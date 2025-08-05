#!/usr/bin/env bash

# GsTaichi release test suite

# Usage: `bash release_test.sh`

# This script is created mainly for eyeball-testing
# that if all of the official examples are still working
# with the latest version of GsTaichi.

# Some of the test cases are fetched from external repositories
# please reach out to us if you are the owner of those
# repos and don't like us to do it.

# You can add more tests into this script and plug-n-play
# existing tests in the `gstaichi::test::main` function as
# you need.

function gstaichi::utils::set_debug {
    if [ ${DEBUG} == "true" ]; then
        set -x
    fi
    set -euo pipefail
}

function gstaichi::utils::logger {
    # default: gray
    if [ "$1" == "info" ]; then
        printf '\e[1;90m%-6s\e[m\n' "$(date +"[%m-%d %H:%M:%S]") $2"
    # error: red
    elif [ "$1" == "error" ]; then
        printf '\e[1;91m%-6s\e[m\n' "$(date +"[%m-%d %H:%M:%S]") $2"
    # success: green
    elif [ "$1" == "success" ]; then
        printf '\e[1;92m%-6s\e[m\n' "$(date +"[%m-%d %H:%M:%S]") $2"
    # warning: yellow
    elif [ "$1" == "warning" ]; then
        printf '\e[1;93m%-6s\e[m\n' "$(date +"[%m-%d %H:%M:%S]") $2"
    # debug: gray
    elif [ "$1" == "debug" ]; then
        if [ "${DEBUG}" == "true" ]; then
            printf '\e[1;90m%-6s\e[m\n' "$(date +"[%m-%d %H:%M:%S]") $2"
        fi
    else
        printf "$1"
    fi
}

function gstaichi::utils::logger::info {
    gstaichi::utils::logger "info" "$1"
}

function gstaichi::utils::logger::error {
    gstaichi::utils::logger "error" "$1"
}

function gstaichi::utils::logger::success {
    gstaichi::utils::logger "success" "$1"
}

function gstaichi::utils::logger::warning {
    gstaichi::utils::logger "warning" "$1"
}

function gstaichi::utils::logger::debug {
    gstaichi::utils::logger "debug" "$1"
}

function gstaichi::utils::line {
    printf '%.0s-' {1..20}; echo
}

function gstaichi::utils::git_clone {
    local GIT_ORG=$1
    local GIT_REPO=$2
    git clone "git@github.com:${GIT_ORG}/${GIT_REPO}.git"
}

function gstaichi::utils::pause {
    read -p "Press enter to continue"
}

function gstaichi::utils::pkill {
    sleep 5
    pkill -f "$1"
}

function gstaichi::test::diffgstaichi {
    local WORKDIR=${1}
    local PATTERN="*.py"
    local ORG="gstaichi-dev"
    local REPO="diffgstaichi"

    # divider
    gstaichi::utils::line
    gstaichi::utils::logger::info "Running DiffGsTaichi examples"

    # clone the repo
    gstaichi::utils::git_clone "${ORG}" "${REPO}"
    cd "${REPO}/examples"

    # run tests
    for match in $(find ./ -name "${PATTERN}"); do
        python "${match}" &
        gstaichi::utils::pkill "${match}"
        gstaichi::utils::line
        # gstaichi::utils::pause
    done

    # go back to workdir
    cd "${WORKDIR}"
}

function gstaichi::test::gstaichi_elements {
    local WORKDIR=${1}
    local PATTERN="demo_*.py"
    local ORG="gstaichi-dev"
    local REPO="gstaichi_elements"

    # divider
    gstaichi::utils::line
    gstaichi::utils::logger::info "Running GsTaichi Elements examples"

    # clone the repo
    gstaichi::utils::git_clone "${ORG}" "${REPO}"
    cd "${REPO}"

    # install dependencies
    python "download_ply.py"


    # run tests
    cd "${REPO}/demo"
    for match in $(find ./ -name "${PATTERN}"); do
        python "${match}" &
        gstaichi::utils::pkill "${match}"
        gstaichi::utils::line
        # gstaichi::utils::pause
    done

    # run special tests
    # FIXME: this does not work properly yet
    # gstaichi::utils::logger::success $(ls)
    # read -p "Please input the directory containing the generated particles, e.g. sim_2022-01-01_20-55-48" particles_dir
    # python render_particles.py -i ./"${particles_dir}" \
    #                            -b 0 -e 400 -s 1 \
    #                            -o ./output \
    #                            --gpu-memory 20 \
    #                            -M 460 \
    #                            --shutter-time 0.0 \
    #                            -r 128

    # go back to workdir
    cd "${WORKDIR}"
}

function gstaichi::test::stannum {
    local WORKDIR=${1}
    local ORG="ifsheldon"
    local REPO="stannum"

    # divider
    gstaichi::utils::line
    gstaichi::utils::logger::info "Running Stannum examples"

    # clone the repo
    gstaichi::utils::git_clone "${ORG}" "${REPO}"
    cd "${REPO}"

    # run tests
    pytest -v -s ./
    gstaichi::utils::line

    # go back to workdir
    cd "${WORKDIR}"
}

function gstaichi::test::sandyfluid {
    local WORKDIR=${1}
    local ORG="ethz-pbs21"
    local REPO="SandyFluid"

    # divider
    gstaichi::utils::line
    gstaichi::utils::logger::info "Running SandyFluid examples"

    # clone the repo
    gstaichi::utils::git_clone "${ORG}" "${REPO}"
    cd "${REPO}"

    # install dependencies
    # remove the line contains pinned GsTaichi version for testing purposes
    grep -v "gstaichi" requirements.txt > tmpfile && mv tmpfile requirements.txt
    pip install -r requirements.txt

    # run tests
    python src/main.py &
    gstaichi::utils::pkill "src/main.py"
    gstaichi::utils::line
    # go back to workdir
    cd "${WORKDIR}"
}

function gstaichi::test::voxel_editor {
    local WORKDIR=${1}
    local ORG="gstaichi-dev"
    local REPO="voxel_editor"

    # divider
    gstaichi::utils::line
    gstaichi::utils::logger::info "Running Voxel Editor examples"

    # clone the repo
    gstaichi::utils::git_clone "${ORG}" "${REPO}"
    cd "${REPO}"

    # run tests
    python voxel_editor.py &
    gstaichi::utils::pkill "voxel_editor.py"
    gstaichi::utils::line

    # go back to workdir
    cd "${WORKDIR}"
}

function gstaichi::test::generate_videos {
    local WORKDIR=${1}
    local PATTERN="test_*.py"
    local ORG="gstaichi-dev"
    local REPO="gstaichi"

    # divider
    gstaichi::utils::line
    gstaichi::utils::logger::info "Generating examples videos"

    # clone the repo
    gstaichi::utils::git_clone "${ORG}" "${REPO}"
    # mkdir "${REPO}/misc/output_videos"

    # run tests
    cd "${REPO}/tests/python/examples"
    for directory in $(find ./ -mindepth 1 -maxdepth 1 -name "*" ! -name "__*" -type d); do
        cd "${directory}"
        for match in $(find ./ -maxdepth 1 -name "${PATTERN}" -type f); do
            pytest -v "${match}"
            gstaichi::utils::line
            # gstaichi::utils::pause
        done
        cd ..
    done

    # go back to workdir
    cd "${WORKDIR}"
}

function gstaichi::test::main {
    # set debugging flag
    DEBUG="false"

    # create a temporary directory for testing
    WORKDIR="$(mktemp -d)"
    gstaichi::utils::logger::info "Running all tests within ${WORKDIR}"

    # make sure to clean up the temp dir on exit
    trap '{ rm -rf -- "$WORKDIR"; }' EXIT

    # walk into the working dir
    cd "${WORKDIR}"

    # diffgstaichi examples
    gstaichi::test::diffgstaichi "${WORKDIR}"

    # gstaichi_elements examples
    gstaichi::test::gstaichi_elements "${WORKDIR}"

    # stannum tests
    gstaichi::test::stannum "${WORKDIR}"

    # sandyfluid tests
    gstaichi::test::sandyfluid "${WORKDIR}"

    # voxel editor tests
    gstaichi::test::voxel_editor "${WORKDIR}"
}

gstaichi::test::main
