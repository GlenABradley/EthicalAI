import sys
import pkg_resources

def check_package(package_name):
    try:
        dist = pkg_resources.get_distribution(package_name)
        print(f"✓ {package_name} {dist.version} is installed")
        return True
    except pkg_resources.DistributionNotFound:
        print(f"✗ {package_name} is NOT installed")
        return False

print(f"Python {sys.version}")
print("Checking required packages...")

required = [
    'fastapi',
    'uvicorn',
    'httpx',
    'pytest',
    'pytest-asyncio',
    'pytest-cov'
]

all_installed = all(check_package(pkg) for pkg in required)

if not all_installed:
    print("\nSome required packages are missing. Please install them with:")
    print("pip install " + " ".join(required))
else:
    print("\nAll required packages are installed!")
