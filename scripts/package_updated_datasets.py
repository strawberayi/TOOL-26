import json
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'UPDATED_SKIN_DATASETS_2026-10-12'
ARCHIVE = ROOT / (NAME + '.zip')
FOLDERS = [
    'Combined_dataset_600',
    'Healthy_skin_300',
    'Combined_dataset_600_YOLO',
    'Combined_dataset_600_reports',
    'A_NEW_DATASETS_support',
]
README = '''Updated skin datasets — 12 October 2026

Upload this entire ZIP to Google Drive. Extract it before using the dataset.

Folders:
Combined_dataset_600: curated old + new disease images, AnyLabeling JSONs,
  and augmented TRAINING / VALIDATION / TESTING folders.
Healthy_skin_300: 300 original body-only healthy skin photos, empty TXT and
  JSON annotations, source metadata, licensing, and visual review sheets.
Combined_dataset_600_YOLO: ready-to-use YOLO images and labels. Use data.yaml
  for the main dataset and stress_test.yaml for the separate stress test.
Combined_dataset_600_reports: disease curation, counts, split, augmentation,
  provenance and verification reports; final healthy counts are in
  healthy_integration_summary.json and the YOLO classes.json.
A_NEW_DATASETS_support: supporting new-dataset curation records.

Logical dataset counts:
4,803 disease examples across eight classes (seven classes have 600;
Impetigo has 603), plus 300 healthy negative examples = 5,103 examples.
Some images occur in multiple folders because these are different exports
of the same dataset. Do not combine these folders as additional examples.

Disease splits: approximately 70% training / 20% validation / 10% testing
per class. Augmentation was applied only to training images.
250 healthy photos (FST I-V): 175 train / 50 validation / 25 main test.
50 healthy FST VI photos: separate stress_test, excluded from main splits.
Final YOLO counts: train 3,537; val 1,011; test 505; stress_test 50.

Healthy images have blank TXT files and are background examples, not a ninth
class. All eight disease IDs remain unchanged. Patient/source groups and
augmentation parents were kept within their assigned splits. Complete
patient independence cannot be established where source IDs are missing.

Healthy source: MSKCC Skin Tone Labeling Dataset, DOI 10.34970/962049,
Memorial Sloan Kettering Cancer Center, CC-BY. Source-recorded Fitzpatrick
types: I-II 50, III-V 200, VI 50. These are clinical close-up body photos;
phototype does not establish Filipino or Southeast Asian ancestry.
Consult the source metadata and attribution records inside each folder.

Current model weights have not been retrained on this updated export.
PACKAGING_MANIFEST.json lists every packaged source file with size and ZIP
CRC32. The completed archive is checked by reading every entry and comparing
the source inventory before reporting success.
'''

if ARCHIVE.exists():
    raise SystemExit(f'Refusing to overwrite: {ARCHIVE}')

files = []
counts = {}
for folder in FOLDERS:
    directory = ROOT / folder
    assert directory.is_dir(), directory
    entries = sorted(p for p in directory.rglob('*') if p.is_file())
    assert all(not p.is_symlink() for p in directory.rglob('*')), directory
    counts[folder] = {'files': len(entries), 'bytes': sum(p.stat().st_size for p in entries)}
    files.extend(entries)

total = sum(item['bytes'] for item in counts.values())
print(json.dumps({'stage': 'start', 'archive': str(ARCHIVE), 'files': len(files), 'source_bytes': total, 'folders': counts}), flush=True)
start = last = time.monotonic()
manifest = []
processed = 0
with zipfile.ZipFile(ARCHIVE, mode='x', compression=zipfile.ZIP_DEFLATED,
                     compresslevel=1, allowZip64=True) as output:
    output.writestr(NAME + '/README_UPLOAD.txt', README)
    for index, path in enumerate(files, 1):
        before = path.stat()
        relative = path.relative_to(ROOT).as_posix()
        member = NAME + '/' + relative
        output.write(path, member)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f'Source changed: {path}'
        info = output.getinfo(member)
        manifest.append({'path': relative, 'bytes': before.st_size, 'crc32': f'{info.CRC:08x}'})
        processed += before.st_size
        now = time.monotonic()
        if now - last >= 20:
            print(json.dumps({'stage': 'compressing', 'files': index, 'total_files': len(files), 'percent_bytes': round(100 * processed / total, 1), 'archive_bytes_so_far': ARCHIVE.stat().st_size, 'elapsed_seconds': round(now-start)}), flush=True)
            last = now
    output.writestr(NAME + '/PACKAGING_MANIFEST.json', json.dumps({'date': '2026-10-12', 'folder_inventory': counts, 'source_files': manifest}, indent=2) + '\n')

print(json.dumps({'stage': 'verifying', 'archive_bytes': ARCHIVE.stat().st_size}), flush=True)
verified = 0
last = time.monotonic()
with zipfile.ZipFile(ARCHIVE) as package:
    expected = {NAME + '/' + item['path'] for item in manifest}
    expected.update({NAME + '/README_UPLOAD.txt', NAME + '/PACKAGING_MANIFEST.json'})
    assert len(package.infolist()) == len(expected)
    assert set(package.namelist()) == expected
    for info in package.infolist():
        with package.open(info) as content:
            while chunk := content.read(1024 * 1024):
                verified += len(chunk)
        now = time.monotonic()
        if now - last >= 20:
            print(json.dumps({'stage': 'verifying', 'percent_bytes': round(100 * verified / total, 1), 'elapsed_seconds': round(now-start)}), flush=True)
            last = now

result = {'status': 'PASS', 'archive': str(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
          'packaged_source_files': len(files), 'archive_entries': len(files) + 2,
          'source_bytes': total, 'crc_verified_every_entry': True,
          'folder_inventory': counts, 'elapsed_seconds': round(time.monotonic()-start)}
(ROOT / (NAME + '_verification.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
