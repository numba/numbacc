name: mlir_bindings_conda_builder

on:
  push:
    branches:
      - main
  pull_request:
    paths:
      - .github/workflows/mlir_conda_builder.yml
      - conda-recipes/mlir-python-bindings/**
  workflow_dispatch:
    inputs:
      platform:
        description: Conda Platform
        default: linux-64
        required: true
        type: choice
        options:
          - all
          - linux-64

# Add concurrency control
concurrency:
  group: >-
    ${{ github.workflow }}-
    ${{ (github.event_name == 'push' && github.ref)
      || github.event.pull_request.number
      || toJson(github.event.inputs)
      || github.sha }}
  cancel-in-progress: true

env:
  ARTIFACT_RETENTION_DAYS: 7

jobs:
  build:
    name: ${{ matrix.platform }}-py${{ matrix.python-version }}-build
    runs-on: ${{ matrix.runner }}
    env:
      EXTRA_CHANNELS: ''
    defaults:
      run:
        shell: bash -elx {0}
    strategy:
      matrix: ${{fromJson(needs.check.outputs.matrix)}}
      fail-fast: false

    steps:
      - name: Clone repository
        uses: actions/checkout@08c6903cd8c0fde910a37f88322edcfb5dd907a8 # v5.0.0
        with:
          fetch-depth: 0

      - name: Setup Miniconda
        uses: conda-incubator/setup-miniconda@835234971496cad1653abb28a638a281cf32541f # v3.2.0
        with:
          auto-update-conda: true
          auto-activate-base: true

      - name: Install conda-build
        run: conda install conda-build

      - name: Build mlir-python-bindings conda package
        run: |
          if [ "${{ inputs.mlir_run_id }}" != "" ]; then
              MLIR_CHANNEL="file:///${{ github.workspace }}/mlir_conda_packages"
          else
              MLIR_CHANNEL="numba/label/dev"
          fi
          CONDA_CHANNEL_DIR="conda_channel_dir"
          mkdir $CONDA_CHANNEL_DIR
          if [ -n "${EXTRA_CHANNELS}" ]; then
            extra_args=(${EXTRA_CHANNELS})
          else
            extra_args=()
          fi
          conda build --debug -c "${MLIR_CHANNEL}" "${extra_args[@]}" -c defaults --python=${{ matrix.python-version }} conda-recipes/mlir-python-bindings --output-folder="${CONDA_CHANNEL_DIR}" --no-test

      - name: Upload mlir-python-bindings conda package
        uses: actions/upload-artifact@330a01c490aca151604b8cf639adc76d48f6c5d4 # v5.0.0
        with:
          name: mlir-python-bindings-${{ matrix.platform }}-py${{ matrix.python-version }}
          path: conda_channel_dir
          compression-level: 0
          retention-days: ${{ env.ARTIFACT_RETENTION_DAYS }}
          if-no-files-found: error

      - name: Show Workflow Run ID
        run: "echo \"Workflow Run ID: ${{ github.run_id }}\""
