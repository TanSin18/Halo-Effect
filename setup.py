"""
Setup script for Halo Effect Analysis package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="halo-effect-analysis",
    version="1.0.0",
    author="SK0759",
    author_email="",
    description="Retail store closure halo effect analysis using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TanSin18/Halo-Effect",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "halo-baby=halo_effect.baby_trade_area:main",
            "halo-bbby=halo_effect.bbby_trade_area:main",
            "halo-closed=halo_effect.closed_stores_trade_area:main",
            "halo-forecast=halo_effect.forecasting_covid_halo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
