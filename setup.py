import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read().splitlines()

setuptools.setup(
    name="ovcontrolnet_tools",
    version="1.0.1",
    author="suk-6",
    author_email="me@suk.kr",
    description="Packages for simple implementations of OVControlNet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/suk-6/ovcontrolnet-tools",
    packages=setuptools.find_packages(
        include=["ovcontrolnet_tools", "ovcontrolnet_tools.*"]
    ),
    install_requires=install_requires,
    python_requires=">=3.6",
    license="MIT",
)
