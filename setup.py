import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ovcontrolnet_tools",
    version="1.0.0",
    author="suk-6",
    author_email="me@suk.kr",
    description="Packages for simple implementations of OVControlNet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/suk-6/ovcontrolnet-tools",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    license="MIT",
)
