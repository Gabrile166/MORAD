#!/usr/bin/env bash
# Download the checkpoints and data used by MORAD.
#
#   bash scripts/download_assets.sh             # everything
#   bash scripts/download_assets.sh ride data   # selected components
#
# Components
#   ride      pretrained RIDE policy (RIDER release)  -> artifacts/models/ride/checkpoint.h5
#   rhofold   RhoFold+ weights                       -> $MORAD_THIRD_PARTY/rhofold_protocol/checkpoints/
#   data      training / test pools and native pairs -> artifacts/pools/
#   morad     MORAD policy after 390 updates         -> artifacts/models/morad/morad_update390.pt
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
third_party_root="${MORAD_THIRD_PARTY:-$(cd "${repo_root}/.." && pwd)/third_party}"
hf_data="${MORAD_HF_DATA:-MORAD-RNA/MORAD-Targets}"
hf_model="${MORAD_HF_MODEL:-MORAD-RNA/MORAD-RIDE}"

RIDE_URL="https://github.com/COLA-Laboratory/RIDER/raw/799f9a0f6587b725c6319dcc24e423e65bcc3682/saved_models/checkpoint.h5"
RIDE_SHA256="09465b6195356a5c76a1e83cd7e2ed86a5972abfda7ae903d3a6025a5283b123"
RHOFOLD_URL="https://huggingface.co/cuhkaih/rhofold/resolve/main/rhofold_pretrained_params.pt"
RHOFOLD_SHA256="3adb621978dfcd7ea0dc0edeb520d249f423d61530df85ff58a1fad2f33e5608"
TRAIN_SHA256="cbb50ebe5375eced320b8b2f15a32f7cf5490e799a06e0133da2fd1019d108d7"
TEST_SHA256="b4d9c5713b3853225f9a1c7a6d2c018a2fc9cb450b9fb71f0258dc6757a8e974"
MORAD_SHA256="dcebb89289181a0946116b899ef4eafd75cd64637b321e93a350e6c12fdb7cbf"

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

fetch() {
    local url="$1" dest="$2" expected="${3:-}"
    mkdir -p "$(dirname "${dest}")"
    if [[ -s "${dest}" ]]; then
        printf 'exists   %s\n' "${dest}"
    else
        printf 'download %s\n' "${dest}"
        curl -L --fail --retry 3 -o "${dest}.part" "${url}"
        mv "${dest}.part" "${dest}"
    fi
    if [[ -n "${expected}" ]]; then
        local actual
        actual="$(sha256 "${dest}")"
        if [[ "${actual}" != "${expected}" ]]; then
            printf 'sha256 mismatch for %s\n  expected %s\n  got      %s\n' "${dest}" "${expected}" "${actual}" >&2
            exit 1
        fi
        printf 'sha256 ok\n'
    fi
}

components=("$@")
if [[ ${#components[@]} -eq 0 ]]; then
    components=(ride rhofold data morad)
fi

for component in "${components[@]}"; do
    case "${component}" in
        ride)
            fetch "${RIDE_URL}" "${repo_root}/artifacts/models/ride/checkpoint.h5" "${RIDE_SHA256}"
            ;;
        rhofold)
            fetch "${RHOFOLD_URL}" \
                "${third_party_root}/rhofold_protocol/checkpoints/rhofold_pretrained_params.pt" \
                "${RHOFOLD_SHA256}"
            ;;
        data)
            data_url="https://huggingface.co/datasets/${hf_data}/resolve/main/data"
            fetch "${data_url}/train.pt" "${repo_root}/artifacts/pools/train527.pt" "${TRAIN_SHA256}"
            fetch "${data_url}/test.pt" "${repo_root}/artifacts/pools/test153.pt" "${TEST_SHA256}"
            fetch "${data_url}/test_native_pairs.json" "${repo_root}/artifacts/pools/test153_native_2d.json"
            ;;
        morad)
            fetch "https://huggingface.co/${hf_model}/resolve/main/morad_ride.pt" \
                "${repo_root}/artifacts/models/morad/morad_update390.pt" "${MORAD_SHA256}"
            ;;
        *)
            printf 'unknown component: %s (expected ride, rhofold, data or morad)\n' "${component}" >&2
            exit 2
            ;;
    esac
done
