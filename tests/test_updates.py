"""Updater: packs, release building, first-run download, upgrades, resume, and the apply helper."""

from __future__ import annotations

import http.server
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest

from cutout.update import client
from cutout.update import packs as P
from cutout.update.manifest import Manifest, ManifestError, is_newer, parse_version

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_release", ROOT / "packaging" / "make_release.py")
make_release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_release)

FILES = {
    "Cutout.exe": b"gui v1",
    "cutout-cli.exe": b"cli v1",
    "_internal/base_library.zip": b"stdlib",
    "_internal/cutout/ui/icon.svg": b"<svg/>",
    "_internal/numpy/_core.pyd": b"numpy" * 100,
    "_internal/old_lib.pyd": b"dropped in v2",
    "_internal/onnxruntime/capi/onnxruntime.dll": b"ort",
    "_internal/onnxruntime/capi/onnxruntime_providers_cuda.dll": b"ort cuda" * 50,
    "_internal/PySide6/Qt6Core.dll": b"qt" * 100,
    "_internal/shiboken6/shiboken6.abi3.dll": b"shiboken",
    "_internal/nvidia_dlls/cudnn64_9.dll": os.urandom(5000),
    "_internal/nvidia_dlls/cudart64_13.dll": b"cudart" * 10,
    "models/manifest.json": b"{}",
    "models/toonout.onnx": os.urandom(7000),
    "models/sam2.1-hiera-base-plus.encoder.onnx": os.urandom(3000),
}


def make_app(root: Path, files=FILES) -> Path:
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root


def release(tmp_path: Path, name: str, files=FILES, previous: Manifest | None = None, part_size=10**9):
    app = make_app(tmp_path / f"{name}-app", files)
    out = tmp_path / f"{name}-release"
    manifest, upload = make_release.build(app, out, name, f"https://example.invalid/{name}", previous,
                                          notes=f"notes for {name}", part_size=part_size)
    manifest.save(app / P.RELEASE_FILE)
    return app, out, manifest, upload


def apply_in_python(plan_path: Path) -> None:
    """What apply_update.ps1 does, minus the waiting, for tests on any OS."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    app = Path(plan["app_dir"])
    for rel in plan["remove"]:
        (app / rel).unlink(missing_ok=True)
    for a in plan["add"]:
        dest = app / a["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(a["src"], dest)


@pytest.fixture(autouse=True)
def _updates_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CUTOUT_UPDATES_DIR", str(tmp_path / "updates"))


# ------------------------------------------------------------------ packs


def test_pack_for_splits_rarely_changing_files_out():
    assert P.pack_for("Cutout.exe") == P.APP
    assert P.pack_for("_internal/cutout/ui/icon.svg") == P.APP
    assert P.pack_for("_internal/base_library.zip") == P.APP
    assert P.pack_for("models/manifest.json") == P.APP
    assert P.pack_for("_internal/numpy/_core.pyd") == P.LIBS
    assert P.pack_for("_internal/onnxruntime/capi/onnxruntime.dll") == P.LIBS
    assert P.pack_for("_internal/onnxruntime/capi/onnxruntime_providers_cuda.dll") == P.GPU
    assert P.pack_for("_internal\\nvidia_dlls\\cudnn64_9.dll") == P.GPU
    assert P.pack_for("_internal/PySide6/Qt6Core.dll") == P.QT
    assert P.pack_for("models/Toonout.onnx") == "model-toonout"
    assert P.pack_for("models/sam2.1-hiera-base-plus.encoder.onnx") == "model-sam2.1-hiera-base-plus.encoder"
    assert P.pack_for("release.json") is None and P.pack_for("install-state.json") is None


@pytest.mark.parametrize("bad", ["../evil.dll", "C:/Windows/x.dll", "a/../../b", "/abs", "", "a/./b"])
def test_safe_rel_rejects_paths_outside_the_install(bad):
    with pytest.raises(ValueError):
        P.safe_rel(bad)


def test_pack_hash_depends_on_names_and_content(tmp_path):
    make_app(tmp_path, {"a": b"1", "b": b"2"})
    h = P.pack_hash(tmp_path, ["a", "b"])
    assert h == P.pack_hash(tmp_path, ["b", "a"])
    (tmp_path / "b").write_bytes(b"3")
    assert P.pack_hash(tmp_path, ["a", "b"]) != h
    assert P.pack_hash(tmp_path, ["a", "missing"]) is None


def test_version_compare():
    assert parse_version("v0.1.57") == (0, 1, 57)
    assert is_newer("0.1.10", "0.1.9") and not is_newer("0.1.9", "0.1.10")
    assert not is_newer("0.1.0", "0.1") and is_newer("0.2", "0.1.99")


def test_manifest_from_a_newer_format_asks_for_new_setup():
    with pytest.raises(ManifestError, match="newer updater"):
        Manifest.from_json(json.dumps({"format": 99, "version": "9", "packs": {}}))
    with pytest.raises(ManifestError, match="damaged"):
        Manifest.from_json("{\"version\": 1}")


# ------------------------------------------------------------- releases


def test_release_has_one_pack_per_kind_and_round_trips(tmp_path):
    app, out, manifest, upload = release(tmp_path, "0.1.1")
    assert set(manifest.packs) == {P.APP, P.LIBS, P.QT, P.GPU, "model-toonout",
                                   "model-sam2.1-hiera-base-plus.encoder"}
    assert Manifest.load(out / "cutout-release.json") == manifest
    for pack in manifest.packs.values():
        assert P.pack_hash(app, pack.files) == pack.hash
        for part in pack.parts:
            assert part.url == f"https://example.invalid/0.1.1/{part.name}"
            assert (out / part.name).stat().st_size == part.size
    assert {p.name for p in upload} == {q.name for pk in manifest.packs.values() for q in pk.parts} | {
        "cutout-release.json"}
    assert all(f.lower() in manifest.file_set() for f in FILES)


def test_big_packs_are_split_under_the_part_size(tmp_path):
    _app, out, manifest, _ = release(tmp_path, "0.1.1", part_size=2000)
    gpu = manifest.packs[P.GPU]
    assert len(gpu.parts) >= 3 and all(p.size <= 2000 for p in gpu.parts)
    joined = b"".join((out / p.name).read_bytes() for p in gpu.parts)
    assert zipfile.ZipFile(__import__("io").BytesIO(joined)).namelist()


def test_unchanged_packs_reuse_the_previous_release_download(tmp_path):
    _a1, _o1, v1, _ = release(tmp_path, "0.1.1")
    v2_files = {**FILES, "Cutout.exe": b"gui v2"}
    _a2, _o2, v2, upload = release(tmp_path, "0.1.2", v2_files, previous=v1)
    assert v2.packs[P.APP].hash != v1.packs[P.APP].hash
    assert v2.packs[P.APP].parts[0].url.startswith("https://example.invalid/0.1.2/")
    for name in (P.LIBS, P.QT, P.GPU, "model-toonout"):
        assert v2.packs[name].parts == v1.packs[name].parts
    assert sorted(p.name for p in upload) == sorted([v2.packs[P.APP].parts[0].name, "cutout-release.json"])


def test_setup_carries_only_app_libs_and_qt(tmp_path):
    app, _out, manifest, _ = release(tmp_path, "0.1.1")
    later = make_release.stage_setup(app, manifest, tmp_path / "setup")
    staged = set(P.list_files(tmp_path / "setup"))
    assert (tmp_path / "setup" / P.RELEASE_FILE).is_file()
    assert "Cutout.exe" in staged and "_internal/PySide6/Qt6Core.dll" in staged
    assert not any(f.startswith(("models/", "_internal/nvidia_dlls/")) and f.endswith((".onnx", ".dll"))
                   for f in staged)
    assert "_internal/onnxruntime/capi/onnxruntime_providers_cuda.dll" not in staged
    assert later == sum(p.download_size for n, p in manifest.packs.items() if n not in P.SETUP_PACKS)


# --------------------------------------------------- first start, upgrade


def test_first_start_downloads_only_what_setup_left_out(tmp_path):
    app, out, manifest, _ = release(tmp_path, "0.1.1")
    install = tmp_path / "install"
    make_release.stage_setup(app, manifest, install)

    status = client.local_status(root=install)
    assert {p.name for p in status.missing} == {P.GPU, "model-toonout", "model-sam2.1-hiera-base-plus.encoder"}
    plan = client.plan_update(status, status.release)
    assert not plan.is_upgrade and plan.remove == []
    seen = []
    adds = client.Downloader(plan, client.work_dir(manifest), mirrors=[out],
                             progress=lambda d, t, w: seen.append((d, t))).run()
    assert seen[-1][0] == seen[-1][1] == plan.download_size
    apply_in_python(client.write_apply_plan(install, plan, adds, client.work_dir(manifest), None, None))

    status = client.local_status(root=install)
    assert status.missing == [] and len(status.have) == len(manifest.packs)
    assert (install / "models/toonout.onnx").read_bytes() == FILES["models/toonout.onnx"]


def test_upgrade_fetches_changed_packs_and_removes_dropped_files(tmp_path):
    app1, out1, v1, _ = release(tmp_path, "0.1.1")
    install = tmp_path / "install"
    shutil.copytree(app1, install)
    (install / "unins000.exe").write_bytes(b"inno uninstaller")
    v2_files = {k: v for k, v in FILES.items() if k != "_internal/old_lib.pyd"}
    v2_files["Cutout.exe"] = b"gui v2"
    _app2, out2, v2, _ = release(tmp_path, "0.1.2", v2_files, previous=v1)

    status = client.local_status(root=install)
    assert status.missing == []
    plan = client.plan_update(status, v2)
    assert plan.is_upgrade
    assert {p.name for p in plan.fetch} == {P.APP, P.LIBS}
    assert plan.remove == ["_internal/old_lib.pyd"]
    adds = client.Downloader(plan, client.work_dir(v2), mirrors=[out2, out1]).run()
    apply_in_python(client.write_apply_plan(install, plan, adds, client.work_dir(v2), None, None))

    status = client.local_status(root=install)
    assert status.version == "0.1.2" and status.missing == []
    assert (install / "Cutout.exe").read_bytes() == b"gui v2"
    assert not (install / "_internal/old_lib.pyd").exists()
    assert (install / "unins000.exe").exists()  # outside the folders packs own


def test_leftover_files_in_owned_folders_are_cleaned_up(tmp_path):
    app, _out, manifest, _ = release(tmp_path, "0.1.1")
    (app / "_internal/stale_from_0.0.9.pyd").write_bytes(b"old")
    (app / "models/retired.onnx").write_bytes(b"old")
    status = client.local_status(root=app)
    plan = client.plan_update(status, manifest)
    assert plan.fetch == []
    assert plan.remove == ["_internal/stale_from_0.0.9.pyd", "models/retired.onnx"]


def test_verified_state_is_reused_until_the_version_changes(tmp_path, monkeypatch):
    app, _out, manifest, _ = release(tmp_path, "0.1.1")
    client.local_status(root=app)
    calls = []
    monkeypatch.setattr(P, "pack_hash", lambda *a, **k: calls.append(a) or None)
    assert client.local_status(root=app).missing == []  # trusted, no hashing
    assert calls == []
    (app / "models/toonout.onnx").unlink()
    assert [p.name for p in client.local_status(root=app).missing] == ["model-toonout"]


def test_a_damaged_download_is_refused(tmp_path, server, monkeypatch):
    base, _handler = server
    monkeypatch.setattr(client.time, "sleep", lambda s: None)
    monkeypatch.setenv("NO_PROXY", "*")
    app = make_app(tmp_path / "app")
    out = tmp_path / "rel"
    manifest, _ = make_release.build(app, out, "0.1.1", f"{base}/rel")
    manifest.save(app / P.RELEASE_FILE)
    install = tmp_path / "install"
    make_release.stage_setup(app, manifest, install)
    plan = client.plan_update(client.local_status(root=install), manifest)
    part = manifest.packs["model-toonout"].parts[0]
    data = bytearray((out / part.name).read_bytes())
    data[-30] ^= 0xFF
    (out / part.name).write_bytes(bytes(data))  # the server now hands out a corrupt file
    with pytest.raises(client.UpdateError, match="damaged"):
        client.Downloader(plan, tmp_path / "work").run()


def test_a_zip_with_unexpected_files_is_refused(tmp_path):
    app, out, manifest, _ = release(tmp_path, "0.1.1")
    pack = manifest.packs["model-toonout"]
    evil = tmp_path / "evil"
    evil.mkdir()
    with zipfile.ZipFile(evil / pack.parts[0].name, "w") as zf:
        zf.writestr("../../outside.dll", b"x")
    fake = client.Part(pack.parts[0].name, "x", (evil / pack.parts[0].name).stat().st_size,
                       P.file_digest(evil / pack.parts[0].name))
    plan = client.Plan(target=manifest, fetch=[client.Pack(pack.name, pack.hash, 1, pack.files, [fake])])
    with pytest.raises(client.UpdateError):
        client.Downloader(plan, tmp_path / "work", mirrors=[evil]).run()
    assert not (tmp_path / "outside.dll").exists()


def test_cancel_stops_the_download(tmp_path):
    app, out, manifest, _ = release(tmp_path, "0.1.1")
    install = tmp_path / "install"
    make_release.stage_setup(app, manifest, install)
    plan = client.plan_update(client.local_status(root=install), manifest)
    with pytest.raises(client.Cancelled):
        client.Downloader(plan, tmp_path / "work", mirrors=[out], cancel=lambda: True).run()


# ------------------------------------------------------------ over HTTP


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves files with Range support; the first request for each file drops halfway."""

    root: Path
    dropped: set

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.root / self.path.lstrip("/")
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if path.suffix != ".json" and path.name not in self.dropped and len(body) > 100:
            self.dropped.add(path.name)
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)


@pytest.fixture
def server(tmp_path):
    handler = type("H", (_RangeHandler,), {"root": tmp_path, "dropped": set()})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", handler
    httpd.shutdown()


def test_download_over_http_resumes_after_a_dropped_connection(tmp_path, server, monkeypatch):
    base, handler = server
    monkeypatch.setattr(client.time, "sleep", lambda s: None)
    monkeypatch.setenv("NO_PROXY", "*")
    app = make_app(tmp_path / "app")
    out = tmp_path / "rel"
    manifest, _ = make_release.build(app, out, "0.1.1", f"{base}/rel")
    manifest.save(app / P.RELEASE_FILE)
    install = tmp_path / "install"
    make_release.stage_setup(app, manifest, install)

    target = client.fetch_manifest(f"{base}/rel/cutout-release.json")
    assert target == manifest
    status = client.local_status(root=install)
    plan = client.plan_update(status, target)
    adds = client.Downloader(plan, tmp_path / "work").run()
    assert handler.dropped  # every big file was cut off once and resumed
    apply_in_python(client.write_apply_plan(install, plan, adds, tmp_path / "work", None, None))
    assert client.local_status(root=install).missing == []


def test_missing_release_gives_a_plain_message(server, monkeypatch):
    base, _ = server
    monkeypatch.setenv("NO_PROXY", "*")
    with pytest.raises(client.UpdateError, match="No release"):
        client.fetch_manifest(f"{base}/nothing/cutout-release.json")


# -------------------------------------------------------- the apply helper


def test_apply_plan_only_names_files_inside_the_install(tmp_path):
    app, out, manifest, _ = release(tmp_path, "0.1.1")
    plan = client.Plan(target=manifest, fetch=[], remove=["../outside.txt"])
    with pytest.raises(ValueError):
        client.write_apply_plan(app, plan, [], tmp_path / "work", None, None)


def test_last_result_is_reported_once(tmp_path):
    client.result_path().parent.mkdir(parents=True)
    client.result_path().write_text(json.dumps({"ok": True, "version": "0.1.2"}), encoding="utf-8")
    (client.updates_dir() / "v0.1.2" / "packs").mkdir(parents=True)
    assert client.take_last_result()["version"] == "0.1.2"
    assert client.take_last_result() is None
    assert not (client.updates_dir() / "v0.1.2").exists()  # downloads cleaned up


def test_helper_script_ships_with_the_package():
    assert (Path(client.__file__).with_name("apply_update.ps1")).is_file()


windows_only = pytest.mark.skipif(sys.platform != "win32", reason="the helper runs Windows PowerShell")


def _run_helper(plan_path: Path) -> dict:
    script = client.apply_script(plan_path.parent)
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                    "-File", str(script), "-PlanPath", str(plan_path)], check=True, timeout=120)
    return json.loads(client.result_path().read_text(encoding="utf-8-sig"))


@windows_only
def test_helper_applies_an_upgrade(tmp_path):
    app1, out1, v1, _ = release(tmp_path, "0.1.1")
    install = tmp_path / "Program Files ü" / "Cutout"
    shutil.copytree(app1, install)
    v2_files = {k: v for k, v in FILES.items() if k != "_internal/old_lib.pyd"}
    v2_files["Cutout.exe"] = b"gui v2"
    _a2, out2, v2, _ = release(tmp_path, "0.1.2", v2_files, previous=v1)
    status = client.local_status(root=install)
    plan = client.plan_update(status, v2)
    work = client.work_dir(v2)
    adds = client.Downloader(plan, work, mirrors=[out2, out1]).run()
    result = _run_helper(client.write_apply_plan(install, plan, adds, work, None, None))
    assert result["ok"] is True, result
    assert (install / "Cutout.exe").read_bytes() == b"gui v2"
    assert not (install / "_internal/old_lib.pyd").exists()
    assert not (work / "backup").exists()
    assert client.local_status(root=install).version == "0.1.2"


@windows_only
def test_helper_rolls_back_when_a_file_cannot_be_moved(tmp_path):
    app1, out1, v1, _ = release(tmp_path, "0.1.1")
    install = tmp_path / "install"
    shutil.copytree(app1, install)
    _a2, out2, v2, _ = release(tmp_path, "0.1.2", {**FILES, "Cutout.exe": b"gui v2"}, previous=v1)
    status = client.local_status(root=install)
    plan = client.plan_update(status, v2)
    work = client.work_dir(v2)
    adds = client.Downloader(plan, work, mirrors=[out2, out1]).run()
    adds.append((work / "does-not-exist.dll", "_internal/zzz.dll"))  # fails after the real files moved
    result = _run_helper(client.write_apply_plan(install, plan, adds, work, None, None, retries=1))
    assert result["ok"] is False and "previous version" in result["message"]
    assert (install / "Cutout.exe").read_bytes() == b"gui v1"
    assert client.local_status(root=install).version == "0.1.1"


# ------------------------------------------------------------------- CLI


def test_cli_version_from_source(capsys, monkeypatch):
    from cutout.cli import main

    monkeypatch.delenv("CUTOUT_APP_DIR", raising=False)
    assert main(["--version"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("cutout ") and "cannot update itself" in out


def test_cli_finishes_setup_from_a_local_folder(tmp_path, monkeypatch, capsys):
    from cutout.cli import main

    app, out, manifest, _ = release(tmp_path, "0.1.1")
    install = tmp_path / "install"
    make_release.stage_setup(app, manifest, install)
    monkeypatch.setenv("CUTOUT_APP_DIR", str(install))
    launched = []
    monkeypatch.setattr(client, "launch_apply", lambda p: launched.append(p) or type("H", (), {"pid": 1})())
    assert main(["--check-updates", "--manifest", str(out / "cutout-release.json")]) == 0
    assert "finish installing 0.1.1" in capsys.readouterr().out
    assert main(["--update", "--manifest", str(install / P.RELEASE_FILE), "--from-dir", str(out)]) == 0
    apply_in_python(launched[0])
    assert client.local_status(root=install).missing == []
    assert main(["--update", "--manifest", str(out / "cutout-release.json")]) == 0
    assert "up to date" in capsys.readouterr().out


# ------------------------------------------- reused packs and the release check


def test_reused_packs_get_no_local_zip_even_if_one_is_lying_around(tmp_path):
    _a1, _o1, v1, _ = release(tmp_path, "0.1.1")
    out2 = tmp_path / "0.1.2-release"
    out2.mkdir()
    stale = out2 / v1.packs[P.GPU].parts[0].name
    stale.write_bytes(b"a rebuild with different bytes under the same name")
    _a2, out2, v2, _ = release(tmp_path, "0.1.2", {**FILES, "Cutout.exe": b"gui v2"}, previous=v1)
    assert v2.packs[P.GPU].parts == v1.packs[P.GPU].parts
    assert not stale.exists()
    assert make_release.verify(v2, out2, remote_size=lambda url: None) == []


def test_zips_are_reproducible(tmp_path):
    make_app(tmp_path / "a")
    make_release.write_zip(tmp_path / "a", list(FILES), tmp_path / "1.zip")
    for f in (tmp_path / "a").rglob("*"):
        if f.is_file():
            os.utime(f, (1_900_000_000, 1_900_000_000))
    make_release.write_zip(tmp_path / "a", list(FILES), tmp_path / "2.zip")
    assert (tmp_path / "1.zip").read_bytes() == (tmp_path / "2.zip").read_bytes()


def test_release_check_catches_mismatched_downloads(tmp_path):
    _app, out, manifest, _ = release(tmp_path, "0.1.1")
    assert make_release.verify(manifest, out) == []
    part = manifest.packs["model-toonout"].parts[0]
    (out / part.name).write_bytes(b"x" * part.size)
    gone = manifest.packs[P.GPU].parts[0]
    (out / gone.name).unlink()
    problems = make_release.verify(manifest, out, remote_size=lambda url: gone.size + 1)
    assert any("model-toonout" in p and "SHA-256" in p for p in problems)
    assert any(gone.name in p and "bytes" in p for p in problems)


def test_a_mismatched_local_copy_is_skipped_for_the_real_download(tmp_path, server, monkeypatch):
    base, _handler = server
    monkeypatch.setattr(client.time, "sleep", lambda s: None)
    monkeypatch.setenv("NO_PROXY", "*")
    app = make_app(tmp_path / "app")
    manifest, _ = make_release.build(app, tmp_path / "rel", "0.1.1", f"{base}/rel")
    manifest.save(app / P.RELEASE_FILE)
    install = tmp_path / "install"
    make_release.stage_setup(app, manifest, install)
    bad = tmp_path / "bad"
    bad.mkdir()
    for pack in manifest.packs.values():
        for part in pack.parts:
            (bad / part.name).write_bytes(b"\0" * part.size)  # right size, wrong bytes
    status = client.local_status(root=install)
    plan = client.plan_update(status, manifest)
    adds = client.Downloader(plan, tmp_path / "work", mirrors=[bad]).run()
    apply_in_python(client.write_apply_plan(install, plan, adds, tmp_path / "work", None, None))
    assert client.local_status(root=install).missing == []
