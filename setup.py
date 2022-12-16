import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cyy_torch_algorithm",
    author="cyy",
    version="0.1",
    author_email="cyyever@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyyever/torch_algorithm",
    packages=[
        "cyy_torch_algorithm",
        "cyy_torch_algorithm/quantization",
        "cyy_torch_algorithm/computation",
        "cyy_torch_algorithm/computation/batch_hvp",
        "cyy_torch_algorithm/computation/sample_gjvp",
        "cyy_torch_algorithm/computation/sample_gvjp",
        "cyy_torch_algorithm/computation/sample_gradient",
        "cyy_torch_algorithm/data_structure",
        "cyy_torch_algorithm/shapely_value",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
