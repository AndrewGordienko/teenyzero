from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    def build_extensions(self):
        standard_flag = "/std:c++17" if self.compiler.compiler_type == "msvc" else "-std=c++17"
        for ext in self.extensions:
            if ext.language != "c++":
                continue
            extra_compile_args = list(getattr(ext, "extra_compile_args", []) or [])
            if standard_flag not in extra_compile_args:
                extra_compile_args.append(standard_flag)
            ext.extra_compile_args = extra_compile_args
        super().build_extensions()

    def run(self):
        try:
            super().run()
        except Exception as exc:
            print(f"[*] Skipping optional native extensions ({exc})")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            print(f"[*] Failed to build optional extension {ext.name} ({exc})")


ext_modules = [
    Extension(
        "teenyzero.native._speedups",
        ["native/speedups.cpp", "native/board.cpp"],
        language="c++",
    ),
]

setup(
    name="teenyzero",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": OptionalBuildExt},
    install_requires=[
        "chess",
        "torch",
        "numpy",
        "flask",
    ],
)
