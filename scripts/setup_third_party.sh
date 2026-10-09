#!/usr/bin/env bash
# Clone the external tools used by MORAD into ../third_party at pinned commits
# and build the two binaries (US-align, EternaFold).
#
#   RiboDiffusion               baseline generator
#   RhoFold, rhofold_protocol   RhoFold+ structure predictor (reward oracle and evaluator)
#   USalign                     C1' TM-score
#   EternaFold                  secondary-structure prediction for the pairing metrics
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"
third_party_root="${MORAD_THIRD_PARTY:-${workspace_root}/third_party}"
mkdir -p "${third_party_root}"

clone_reference() {
    local name="$1"
    local url="$2"
    local commit="$3"
    local target="${third_party_root}/${name}"

    local cloned=0
    if [[ ! -d "${target}/.git" ]]; then
        git clone "${url}" "${target}"
        cloned=1
    fi
    git -C "${target}" cat-file -e "${commit}^{commit}"
    if [[ "${cloned}" -eq 1 ]]; then
        git -C "${target}" checkout --detach "${commit}"
    fi

    local current
    current="$(git -C "${target}" rev-parse HEAD)"
    if [[ "${current}" != "${commit}" ]]; then
        printf 'warning: %s is at %s (pinned reference is %s); existing worktree was not changed\n' \
            "${name}" "${current}" "${commit}" >&2
    fi
    printf '%s %s\n' "${name}" "${current}"
}

clone_reference \
    RiboDiffusion \
    https://github.com/ml4bio/RiboDiffusion.git \
    3ac7a557f470c25d95379acedf75a9a49f70ef6e
clone_reference \
    RhoFold \
    https://github.com/ml4bio/RhoFold.git \
    6bdfbda720184409eb682ce08c05d258162ddc48
clone_reference \
    rhofold_protocol \
    https://github.com/WangJiuming/rhofold_protocol.git \
    03ecba600d01fe545e8beb90f40ecb01e12b29bf
clone_reference \
    USalign \
    https://github.com/pylelab/USalign.git \
    fa4376bd99fa17a123d05d7ea47cf6574c80d64f
clone_reference \
    EternaFold \
    https://github.com/eternagame/EternaFold.git \
    87b9aac55cee14fd562049d08f7b92d3131f10ce

if [[ ! -x "${third_party_root}/USalign/USalign" ]]; then
    make -C "${third_party_root}/USalign" USalign
fi
if [[ ! -x "${third_party_root}/EternaFold/src/contrafold" ]]; then
    make -C "${third_party_root}/EternaFold/src"
fi

printf 'third_party %s\n' "${third_party_root}"
printf 'USalign     %s\n' "${third_party_root}/USalign/USalign"
printf 'EternaFold  %s\n' "${third_party_root}/EternaFold/src/contrafold"
printf '%s\n' "Next: bash scripts/download_assets.sh (RIDE checkpoint, RhoFold+ weights, data, MORAD checkpoint)."
