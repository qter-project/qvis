{
  inputs = {
    nixpkgs.url = "nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    qter = {
      url = "github:qter-project/qter";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      rust-overlay,
      qter,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };

        rust = (pkgs.rust-bin.fromRustupToolchainFile qter.toolchain."${system}").override {
          extensions = [
            "rust-src"
            "rust-analyzer"
          ];
        };
      in
      {
        devShell = pkgs.mkShell rec {
          buildInputs =
            (with pkgs; [
              sccache
              rust-analyzer
              rust
              pkg-config
              qter.packages."${system}".shiroa
            ]);

          RUST_BACKTRACE = 1;
          RUSTC_WRAPPER = "sccache";
          SCCACHE_SERVER_PORT = "54226";
          RUSTFLAGS = "-C target-cpu=native";

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;

          shellHook = ''
            export PATH=$PATH:~/.cargo/bin
          '';
        };
      }
    );
}
