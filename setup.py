from setuptools import setup, find_packages
import pathlib

# Root directory
here = pathlib.Path(__file__).parent.resolve()

# Long description from README
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="digital_pathology",
    version="1.0.0",
    author="Israel Llorens",
    author_email="sanchezis@hotmail.com",
    description="Digital Pathology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sanchezis/MU-Big-Data-and-Data-Science-Image-Pathology-Pipeline",  # Update with your repo
    license="EUPL-1.2",
    classifiers=[
        "License :: OSI Approved :: European Union Public Licence 1.2 (EUPL 1.2)",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_dir={"": "dist"},
    packages=find_packages(where="digital_pathology"),
    python_requires=">=3.10, <3.12",
    install_requires=[
    # Cross-platform
    "numpy>=1.24,<2.0",
    "pandas>=2.0.0,<3.0",
    "pyspark>=3.4.4,<4.0",
    "pyarrow>=14.0.2,<15.0",
    "boto3>=1.37.18,<2.0",
    "matplotlib>=3.10.1,<4.0",
    "seaborn>=0.13.2,<0.14",
    "tifffile>=2025.3.13,<2026",
    "tiffslide>=2.5.0,<3.0",
    "slideio>=2.7.0,<3.0",
    "plotly>=6.0.1,<7.0",
    "scipy>=1.8.0,<2.0",
    "scikit-learn>=1.6.1,<2.0",
    "shapely>=2.0.7,<3.0",
    "ipykernel>=6.29.5,<7.0",
    "pillow>=11.1.0,<12.0",
    "statsmodels>=0.14.4,<0.15",
    "opencv-python>=4.11.0.86,<5.0",
    "torch>=2.6.0,<3.0",
    "torchvision>=0.21.0,<0.22",
    "torchaudio>=2.6.0,<3.0",

    # Platform-specific
    "openslide-python>=1.3.1,<2.0; sys_platform != 'win32'",
    "openslide-bin>=4.0.0.6,<5.0; sys_platform != 'win32'",
    
    # Windows-specific
    "tensorflow>=2.16.1,<3.0; sys_platform == 'win32'",
    
    # macOS-specific
    "tensorflow-macos>=2.16.2,<3.0; sys_platform == 'darwin'",
    "tensorflow-metal>=1.2.0,<2.0; sys_platform == 'darwin'",
    
    # Platform-agnostic with version constraints
    "scikit-image>=0.25.2,<0.26",
    "stardist>=0.9.1,<0.10",
    "histomicstk>=1.4.0,<2.0"
    ],
    extras_require={
        "dev": ["mypy>=1.6.1,<2.0", "black>=23.10.0,<24.0"],
        "test": ["pytest>=7.4.4,<8.0", "pytest-cov>=4.1.0,<5.0"],
        'cuda': ["torch>=2.6.0,<3.0 -f https://download.pytorch.org/whl/cu121"],
        'full': [
            "openslide-python>=1.3.1,<2.0",
            "openslide-bin>=4.0.0.6,<5.0; sys_platform != 'win32'"
        ],        
    },
    package_data={
        "digital_pathology": [
            "LICENSE",
            "AUTHORS",
        ]
    },
    entry_points={
        "console_scripts": [
            "dpath=digital_pathology.cli:main",  # Example CLI entry point
        ],
    },
)