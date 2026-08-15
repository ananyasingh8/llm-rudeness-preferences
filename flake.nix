{
  description = "Reproducible Python dev environment with CUDA/NVIDIA GPU support";

  # ============================================================
  # INPUTS
  # ============================================================

  inputs = rec {
    nixpkgs-stable.url = "github:NixOS/nixpkgs/nixos-26.05";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs = nixpkgs-stable;
    flake-utils.url = "github:numtide/flake-utils";
  };

  # ============================================================
  # OUTPUTS
  # ============================================================

  outputs =
    inputs@{ self
    , nixpkgs
    , nixpkgs-stable
    , nixpkgs-unstable
    , flake-utils
    , ...
    }:
    let
      # ==========================================================
      # PROJECT CONFIGURATION — edit this section for your project
      # ==========================================================

      # Python version (nixpkgs attribute name)
      pythonAttr = "python312";

      # Python packages available via `import` in the interpreter
      pythonPackages = ps: with ps; [
        pip
        virtualenv
        tkinter
        # Add your packages here: numpy, pandas, requests, etc.
      ];

      # CLI tools available in $PATH
      cliTools = pkgs_: [
        pkgs_.uv                                    # Package installer
        pkgs_.mypy                                  # Type checker
        pkgs_.${pythonAttr + "Packages"}.ruff       # Linter/formatter
      ];

      # Native build dependencies (C libraries, compilers)
      nativeBuildDeps = pkgs_: with pkgs_; [
        zlib
        glibc
        stdenv.cc.cc.lib
        gcc
        tk
        tcl
        libxcrypt
      ];

      # CUDA build dependencies (GPU libraries, graphics, multimedia)
      cudaBuildDeps = pkgs_: with pkgs_; [
        ffmpeg
        fmt.dev
        libGLU
        libGL
        libxi
        libxmu
        freeglut
        libxext
        libx11
        libxv
        libxrandr
        zlib
        ncurses
        stdenv.cc
        binutils
        wayland
      ];

      # ==========================================================
      # IMPLEMENTATION — you shouldn't need to edit below here
      # ==========================================================

      cudaShellHook = pkgs_: with pkgs_; ''
        export CMAKE_PREFIX_PATH="${pkgs_.fmt.dev}:$CMAKE_PREFIX_PATH"
        export PKG_CONFIG_PATH="${pkgs_.fmt.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
        export EXTRA_CCFLAGS="-I/usr/include"

        # Triton otherwise assumes the FHS-only /sbin/ldconfig path.
        if [[ -d /run/opengl-driver/lib ]]; then
          export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
        fi
      '';

      mkEnvFromChannel = (nixpkgs-channel:
        flake-utils.lib.eachDefaultSystem (system:
          let
            # ----------------------------------------------------------
            # Package Set Configuration
            # ----------------------------------------------------------

            pkgs = import nixpkgs-channel {
              inherit system;
              config.allowUnfree = true;
              config.cudaSupport = true;
              config.cudaVersion = "13";
            };

            # ----------------------------------------------------------
            # Python Environment
            # ----------------------------------------------------------

            pythonWithPkgs = pkgs.${pythonAttr}.withPackages pythonPackages;

            pythonWrapper = pkgs.writeShellScriptBin "python3" ''
              export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
              exec ${pythonWithPkgs}/bin/python3 "$@"
            '';

            # ----------------------------------------------------------
            # Development Shell
            # ----------------------------------------------------------

            devShell = pkgs.mkShell {
              name = "python-cuda-dev"; # <-- Rename to your project

              buildInputs =
                (nativeBuildDeps pkgs)
                ++ (cudaBuildDeps pkgs);

              packages = [ pythonWrapper ] ++ (cliTools pkgs);

              shellHook =
                let
                  tk = pkgs.tk;
                  tcl = pkgs.tcl;
                in
                ''
                  export LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
                  export TK_LIBRARY="${tk}/lib/${tk.libPrefix}"
                  export TCL_LIBRARY="${tcl}/lib/${tcl.libPrefix}"

                  ${cudaShellHook pkgs}

                  if [[ -d .venv ]]; then
                    VENV_PYTHON="$(readlink -f ./.venv/bin/python)"
                    echo "Found .venv/bin/python: $VENV_PYTHON"
                    export PYTHONPATH=".venv/lib/python3.12/site-packages:$PYTHONPATH"
                  fi
                '';

              allowSubstitutes = false;
            };

            # ----------------------------------------------------------
            # FHS Environment
            # For packages requiring traditional Linux filesystem layout
            # ----------------------------------------------------------

            fhsEnv = (pkgs.buildFHSEnv {
              name = "python-cuda-fhs-dev"; # <-- Rename to your project

              targetPkgs = (fhs-pkgs:
                let
                  fhsPython = fhs-pkgs.${pythonAttr}.withPackages pythonPackages;
                in
                [ fhsPython fhs-pkgs.git ]
                ++ (cliTools fhs-pkgs)
                ++ (nativeBuildDeps fhs-pkgs)
                ++ (cudaBuildDeps fhs-pkgs)
              );

              multiPkgs = fhs-pkgs: with fhs-pkgs; [
                zlib
                libxcrypt-legacy
              ];

              # profile uses the outer pkgs — same package set as fhs-pkgs.
              profile = ''
                export LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
                export EXTRA_CCFLAGS="-I/usr/include"
                ${cudaShellHook pkgs}
              '';

              allowSubstitutes = false;
            }).env;

          in
          {
            devShells.default = devShell;
            devShells.build = fhsEnv;
          }
        ));
    in
    mkEnvFromChannel nixpkgs-stable;
}
