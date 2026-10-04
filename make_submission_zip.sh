#!/bin/bash
# Creates the final submission ZIP on your Desktop and checks it.
# Run from the project folder:  bash make_submission_zip.sh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
PARENT_DIR="$(dirname "$PROJECT_DIR")"
OUT="$HOME/Desktop/${PROJECT_NAME}.zip"
LIMIT_BYTES=100000000   # strict 100 MB (decimal), to be safe

if [ ! -f "$PROJECT_DIR/executable/PhishingURLDetector.exe" ]; then
  echo "WARNING: executable/PhishingURLDetector.exe is missing."
  echo "         The college needs the EXE inside the ZIP. Build it with GitHub Actions first."
  echo
fi

rm -f "$OUT"
cd "$PARENT_DIR"
zip -r -q "$OUT" "$PROJECT_NAME" \
  -x "$PROJECT_NAME/.venv/*" \
  -x "$PROJECT_NAME/.git/*" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x "*.DS_Store" \
  -x "$PROJECT_NAME/.vscode/*" \
  -x "$PROJECT_NAME/.idea/*" \
  -x "$PROJECT_NAME/build/*" \
  -x "$PROJECT_NAME/dist/*" \
  -x "*.spec" \
  -x "$PROJECT_NAME/*.zip"

SIZE_BYTES=$(stat -f%z "$OUT")
echo "ZIP created: $OUT"
echo "ZIP size (measured): $SIZE_BYTES bytes  =  $(ls -lh "$OUT" | awk '{print $5}')"
if [ "$SIZE_BYTES" -lt "$LIMIT_BYTES" ]; then
  echo "SIZE CHECK: PASS (below 100,000,000 bytes)"
else
  echo "SIZE CHECK: FAIL (too big - do NOT submit)"
fi

echo
echo "Unwanted files inside the ZIP (should be 0):"
unzip -Z1 "$OUT" | grep -c -E '/\.venv/|/__pycache__/|/\.git/|\.pyc$|\.DS_Store$' || true
echo
echo "EXE inside the ZIP:"
unzip -Z1 "$OUT" | grep -i '\.exe$' || echo "  NONE - the EXE is missing"
echo
echo "Full list of files in the ZIP:"
unzip -l "$OUT"
