{
  description = "Dev shell for torch 2.0.0 and torchvision 0.15.1";
  inputs = {
    nixpkgs-torch.url =
      "github:NixOS/nixpkgs/0da597302cd18b88e9f6f8242a2c9dc6fa9891a5";
    nixpkgs-torchvision.url =
      "github:NixOS/nixpkgs/b82dbe2fdc61a78f9eb931170bfea09fa824888d";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs-torch, nixpkgs-torchvision, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs-torch = import nixpkgs-torch { inherit system; };
      pkgs-torchvision = import nixpkgs-torchvision { inherit system; };
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pkgs.python3
          pkgs-torch.python3Packages.torch
          pkgs-torchvision.python3Packages.torchvision
          pkgs.python3Packages.pandas
          pkgs.python3Packages.numpy
          pkgs.python3Packages.matplotlib
        ];
      };
    };
}
