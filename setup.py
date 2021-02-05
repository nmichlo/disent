import setuptools


with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

with open('requirements.txt', 'r') as f:
    install_requires = (req[0] for req in map(lambda x: x.split('#'), f.readlines()))
    install_requires = [req for req in map(str.strip, install_requires) if req]


setuptools.setup(
    name="disent",
    author="Nathan Juraj Michlo",
    author_email="NathanJMichlo@gmail.com",

    version="0.0.1.dev1",
    python_requires="==3.8",
    packages=setuptools.find_packages(),

    install_requires=install_requires,

    url="https://github.com/nmichlo/eunomia",
    description="Vae disentanglement framework built with pytorch lightning.",
    long_description=long_description,
    long_description_content_type="text/markdown",

    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
    ],
)
