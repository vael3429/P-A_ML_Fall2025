{
  description = "Dev shell with pinned torch/torchvision via mach-nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    mach-nix.url = "github:DavHau/mach-nix";
  };

  outputs = { self, nixpkgs, mach-nix }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          # Create a Python environment with torch/torchvision pinned
          mach-nix.mkPython {
            python = "python310";  # or python311
            requirements = ''
              torch==2.0.0
              torchvision==0.15.1
              numpy
              pandas
              scipy
              matplotlib
              pillow
              ipython
            '';
          }
        ];
      };
    };
}

