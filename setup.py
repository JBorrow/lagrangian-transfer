import setuptools

setuptools.setup(
    name="ltcaesar",
    version="0.3.0",
    description="Library for studying transfer between lagrangian regions using Caesar-yt",
    url="https://github.com/JBorrow/lagrangian-transfer",
    author="Josh Borrow, Daniel Angles-Alcazar",
    author_email="joshua.borrow@durham.ac.uk",
    packages=["ltcaesar"],
    scripts=["analyse.py"],
    zip_safe=False
)
