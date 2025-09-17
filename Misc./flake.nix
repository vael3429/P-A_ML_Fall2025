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
      devShells.${system}.default = pkgs-torch.mkShell {
        buildInputs = [
          (pkgs-torch.python3.withPackages (ps: [
            ps.torch
            pkgs-torchvision.python3Packages.torchvision
            ps.pandas
            ps.numpy
            ps.matplotlib
            ps.pillow
            ps.ipython
          ]))
        ];
      };
    };
}
