# This file is automatically generated by upgrade.py.

ARCHIVES = [
    dict(
        name = "rust_darwin_aarch64__aarch64-apple-darwin__stable",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_darwin_aarch64__aarch64-apple-darwin__stable.bazel"),
        downloads = "[]",
    ),
    dict(
        name = "rust_darwin_aarch64__aarch64-apple-darwin__stable_tools",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_darwin_aarch64__aarch64-apple-darwin__stable_tools.bazel"),
        downloads = json.encode(
            [
                {
                    "sha256": "0c9b1a24f08f7b7eeb411a4a62e2d8f4dbc07af7b26f93306b1c3b5d7abc0a3a",
                    "stripPrefix": "rustc-1.75.0-aarch64-apple-darwin/rustc",
                    "url": [
                        "https://static.rust-lang.org/dist/rustc-1.75.0-aarch64-apple-darwin.tar.gz",
                    ],
                },
                {
                    "sha256": "0f1f20e1a1e3a7c44658a82b5defa8b38fac8c9b4e25051a73350ec36a3472c3",
                    "stripPrefix": "clippy-1.75.0-aarch64-apple-darwin/clippy-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/clippy-1.75.0-aarch64-apple-darwin.tar.gz",
                    ],
                },
                {
                    "sha256": "16eac1143417207620654606f09e575bbdb66014bbd571e89182a4e4f630a3a1",
                    "stripPrefix": "cargo-1.75.0-aarch64-apple-darwin/cargo",
                    "url": [
                        "https://static.rust-lang.org/dist/cargo-1.75.0-aarch64-apple-darwin.tar.gz",
                    ],
                },
                {
                    "sha256": "a003f243ed50cbbd5f8add4838cd479e139b560e7a74669fdfbbaa2776b31492",
                    "stripPrefix": "rustfmt-1.75.0-aarch64-apple-darwin/rustfmt-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/rustfmt-1.75.0-aarch64-apple-darwin.tar.gz",
                    ],
                },
                {
                    "sha256": "400a442a891b8a6ce89a9ac242651b4f032d1a8419cb23874f747f54a4cc9069",
                    "stripPrefix": "llvm-tools-1.75.0-aarch64-apple-darwin/llvm-tools-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/llvm-tools-1.75.0-aarch64-apple-darwin.tar.gz",
                    ],
                },
                {
                    "sha256": "8eedd403d05829369e3dd84c6815f69fb7e5495d3ee3bf2b4b2f04d8591fe639",
                    "stripPrefix": "rust-std-1.75.0-aarch64-apple-darwin/rust-std-aarch64-apple-darwin",
                    "url": [
                        "https://static.rust-lang.org/dist/rust-std-1.75.0-aarch64-apple-darwin.tar.gz",
                    ],
                },
            ],
        ),
    ),
    dict(
        name = "rust_darwin_x86_64__x86_64-apple-darwin__stable",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_darwin_x86_64__x86_64-apple-darwin__stable.bazel"),
        downloads = "[]",
    ),
    dict(
        name = "rust_darwin_x86_64__x86_64-apple-darwin__stable_tools",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_darwin_x86_64__x86_64-apple-darwin__stable_tools.bazel"),
        downloads = json.encode(
            [
                {
                    "sha256": "ed2c9bbae4bda6d89c091a7b32c60655358ec1ade58677eaaa0e5e16ec4fb163",
                    "stripPrefix": "rustc-1.75.0-x86_64-apple-darwin/rustc",
                    "url": [
                        "https://static.rust-lang.org/dist/rustc-1.75.0-x86_64-apple-darwin.tar.gz",
                    ],
                },
                {
                    "sha256": "9bd6ca5d265765837627191f16a70baac4c133de0ee74f0cd0706a2e7e09dd1d",
                    "stripPrefix": "clippy-1.75.0-x86_64-apple-darwin/clippy-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/clippy-1.75.0-x86_64-apple-darwin.tar.gz",
                    ],
                },
                {
                    "sha256": "c54a64ce2e7b6d143e10d3ebd18ab8d41783558b1d9706fded1d75a2826a3463",
                    "stripPrefix": "cargo-1.75.0-x86_64-apple-darwin/cargo",
                    "url": [
                        "https://static.rust-lang.org/dist/cargo-1.75.0-x86_64-apple-darwin.tar.gz",
                    ],
                },
                {
                    "sha256": "2078ea59e09b57e02cde3f54ef015145d7c358c771e555cb2e27d104bf4b462b",
                    "stripPrefix": "rustfmt-1.75.0-x86_64-apple-darwin/rustfmt-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/rustfmt-1.75.0-x86_64-apple-darwin.tar.gz",
                    ],
                },
                {
                    "sha256": "da86f3871300123eddca5e833bbcd6a84d17379e56e7eca5a96bb3e10d57c0ae",
                    "stripPrefix": "llvm-tools-1.75.0-x86_64-apple-darwin/llvm-tools-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/llvm-tools-1.75.0-x86_64-apple-darwin.tar.gz",
                    ],
                },
                {
                    "sha256": "65098155333de2e446df61cdaf12a0c441358b7973f3cb1ba95fd11bda890406",
                    "stripPrefix": "rust-std-1.75.0-x86_64-apple-darwin/rust-std-x86_64-apple-darwin",
                    "url": [
                        "https://static.rust-lang.org/dist/rust-std-1.75.0-x86_64-apple-darwin.tar.gz",
                    ],
                },
            ],
        ),
    ),
    dict(
        name = "rust_linux_aarch64__aarch64-unknown-linux-gnu__stable",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_linux_aarch64__aarch64-unknown-linux-gnu__stable.bazel"),
        downloads = "[]",
    ),
    dict(
        name = "rust_linux_aarch64__aarch64-unknown-linux-gnu__stable_tools",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_linux_aarch64__aarch64-unknown-linux-gnu__stable_tools.bazel"),
        downloads = json.encode(
            [
                {
                    "sha256": "b7da2133a86e15a03e49307c0e91b0ab39c6ec8d0735a1c609499713f7e31571",
                    "stripPrefix": "rustc-1.75.0-aarch64-unknown-linux-gnu/rustc",
                    "url": [
                        "https://static.rust-lang.org/dist/rustc-1.75.0-aarch64-unknown-linux-gnu.tar.gz",
                    ],
                },
                {
                    "sha256": "6c17c7f615da9426ea85468ef7a4c0f13416f96aaf5603854c099c42c42dccbb",
                    "stripPrefix": "clippy-1.75.0-aarch64-unknown-linux-gnu/clippy-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/clippy-1.75.0-aarch64-unknown-linux-gnu.tar.gz",
                    ],
                },
                {
                    "sha256": "8734060ba397ce0306e6b70253551eb63af6982c19326fd734f60ca35814ad9b",
                    "stripPrefix": "cargo-1.75.0-aarch64-unknown-linux-gnu/cargo",
                    "url": [
                        "https://static.rust-lang.org/dist/cargo-1.75.0-aarch64-unknown-linux-gnu.tar.gz",
                    ],
                },
                {
                    "sha256": "a5ad8ae8c975300c35ae1e367c9487401c8b92ef41a218258270dae75e4be885",
                    "stripPrefix": "rustfmt-1.75.0-aarch64-unknown-linux-gnu/rustfmt-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/rustfmt-1.75.0-aarch64-unknown-linux-gnu.tar.gz",
                    ],
                },
                {
                    "sha256": "d4ed2d82c815857397c201491355f6c2a90aad586ceee55421425002ca4345cf",
                    "stripPrefix": "llvm-tools-1.75.0-aarch64-unknown-linux-gnu/llvm-tools-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/llvm-tools-1.75.0-aarch64-unknown-linux-gnu.tar.gz",
                    ],
                },
                {
                    "sha256": "74960aa36e66541b3e9a8f78aa5df9c5c5a0e93207c0bb42a4fa141bccfbfd14",
                    "stripPrefix": "rust-std-1.75.0-aarch64-unknown-linux-gnu/rust-std-aarch64-unknown-linux-gnu",
                    "url": [
                        "https://static.rust-lang.org/dist/rust-std-1.75.0-aarch64-unknown-linux-gnu.tar.gz",
                    ],
                },
            ],
        ),
    ),
    dict(
        name = "rust_linux_x86_64__x86_64-unknown-linux-gnu__stable",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_linux_x86_64__x86_64-unknown-linux-gnu__stable.bazel"),
        downloads = "[]",
    ),
    dict(
        name = "rust_linux_x86_64__x86_64-unknown-linux-gnu__stable_tools",
        build_file = Label("@drake//tools/workspace/rust_toolchain:lock/details/BUILD.rust_linux_x86_64__x86_64-unknown-linux-gnu__stable_tools.bazel"),
        downloads = json.encode(
            [
                {
                    "sha256": "9684bc337f41110821fc94498e0596f76a061fae4667b195579b03fd141ad538",
                    "stripPrefix": "rustc-1.75.0-x86_64-unknown-linux-gnu/rustc",
                    "url": [
                        "https://static.rust-lang.org/dist/rustc-1.75.0-x86_64-unknown-linux-gnu.tar.gz",
                    ],
                },
                {
                    "sha256": "e02b66c86cc55ba4f9c1bcbd75c06dc4387445629d182b67bc5b54431bbd538d",
                    "stripPrefix": "clippy-1.75.0-x86_64-unknown-linux-gnu/clippy-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/clippy-1.75.0-x86_64-unknown-linux-gnu.tar.gz",
                    ],
                },
                {
                    "sha256": "ccd5f13a3101efadf09b1bbbebe8f099d97e99e1d4f0a29a37814a0dae429ede",
                    "stripPrefix": "cargo-1.75.0-x86_64-unknown-linux-gnu/cargo",
                    "url": [
                        "https://static.rust-lang.org/dist/cargo-1.75.0-x86_64-unknown-linux-gnu.tar.gz",
                    ],
                },
                {
                    "sha256": "ee077bf5ead714e7cd2b94babc6673b7d91e485f7c1583441a8d8f701d4b5c6e",
                    "stripPrefix": "rustfmt-1.75.0-x86_64-unknown-linux-gnu/rustfmt-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/rustfmt-1.75.0-x86_64-unknown-linux-gnu.tar.gz",
                    ],
                },
                {
                    "sha256": "3efa2c600c2b5a6493975c4bf6d8d52fb7ddba3c131dbf84363da4e049ce9ed7",
                    "stripPrefix": "llvm-tools-1.75.0-x86_64-unknown-linux-gnu/llvm-tools-preview",
                    "url": [
                        "https://static.rust-lang.org/dist/llvm-tools-1.75.0-x86_64-unknown-linux-gnu.tar.gz",
                    ],
                },
                {
                    "sha256": "b7a43ed4bc9a9205b3ee2ece2a38232c8da5f1f14e7ed84fbefd492f9d474579",
                    "stripPrefix": "rust-std-1.75.0-x86_64-unknown-linux-gnu/rust-std-x86_64-unknown-linux-gnu",
                    "url": [
                        "https://static.rust-lang.org/dist/rust-std-1.75.0-x86_64-unknown-linux-gnu.tar.gz",
                    ],
                },
            ],
        ),
    ),
]
