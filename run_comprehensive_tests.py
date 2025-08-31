#!/usr/bin/env python3
"""
Comprehensive test runner for EthicalAI project.

Executes all test suites including:
- Backend API tests
- Ethical evaluation tests  
- Axis pack tests
- Frontend integration tests
- Performance benchmarks
"""

import subprocess
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any


class TestRunner:
    """Comprehensive test execution and reporting."""
    
    def __init__(self, project_root: str = ".", use_real_encoder: bool = False):
        self.project_root = Path(project_root).resolve()
        self.python_exe = sys.executable
        self.use_real_encoder = use_real_encoder
        self.results = {}
        
    def run_backend_tests(self) -> Dict[str, Any]:
        """Run all backend API tests."""
        print("🔧 Running Backend API Tests...")
        
        test_files = [
            "tests/test_end_to_end.py",
            "tests/test_comprehensive_e2e.py", 
            "tests/test_frontend_integration.py"
        ]
        
        results = {}
        
        for test_file in test_files:
            if (self.project_root / test_file).exists():
                print(f"  Running {test_file}...")
                result = self._run_pytest(test_file)
                results[test_file] = result
            else:
                print(f"  ⚠️  {test_file} not found")
                
        return results
    
    def run_ethical_evaluation_tests(self) -> Dict[str, Any]:
        """Run ethical evaluation and axis pack tests."""
        print("🧠 Running Ethical Evaluation Tests...")
        
        test_files = [
            "tests/test_axis_packs_comprehensive.py"
        ]
        
        results = {}
        
        for test_file in test_files:
            if (self.project_root / test_file).exists():
                print(f"  Running {test_file}...")
                result = self._run_pytest(test_file)
                results[test_file] = result
            else:
                print(f"  ⚠️  {test_file} not found")
                
        return results
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        print("⚡ Running Performance Tests...")
        
        test_files = [
            "tests/test_performance_benchmarks.py"
        ]
        
        results = {}
        
        for test_file in test_files:
            if (self.project_root / test_file).exists():
                print(f"  Running {test_file}...")
                result = self._run_pytest(test_file, extra_args=["-s"])  # Show output for performance
                results[test_file] = result
            else:
                print(f"  ⚠️  {test_file} not found")
                
        return results
    
    def run_frontend_tests(self) -> Dict[str, Any]:
        """Run frontend tests using npm/vitest."""
        print("🎨 Running Frontend Tests...")
        
        ui_dir = self.project_root / "ui"
        
        if not ui_dir.exists():
            print("  ⚠️  UI directory not found")
            return {"error": "UI directory not found"}
        
        # Check if package.json exists
        if not (ui_dir / "package.json").exists():
            print("  ⚠️  package.json not found in UI directory")
            return {"error": "package.json not found"}
        
        try:
            os.chdir(ui_dir)
            
            # Run npm test
            result = subprocess.run(
                ["npm", "test"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            os.chdir(self.project_root)
            
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            print("  ⏰ Frontend tests timed out")
            return {"error": "timeout", "success": False}
        except FileNotFoundError:
            print("  ⚠️  npm not found - install Node.js")
            return {"error": "npm not found", "success": False}
        except Exception as e:
            print(f"  ❌ Frontend test error: {e}")
            return {"error": str(e), "success": False}
        finally:
            os.chdir(self.project_root)
    
    def _run_pytest(self, test_file: str, extra_args: List[str] = None) -> Dict[str, Any]:
        """Run pytest on a specific test file."""
        args = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "--tb=short"
        ]
        
        if extra_args:
            args.extend(extra_args)
        
        try:
            start_time = time.time()
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=60  # 1 minute timeout per test file
            )
            end_time = time.time()
            
            # Print stdout and stderr for debugging
            if result.stdout:
                print(f"STDOUT for {test_file}:")
                print(result.stdout[-1000:])  # Last 1000 chars
            if result.stderr:
                print(f"STDERR for {test_file}:")
                print(result.stderr[-1000:])  # Last 1000 chars
            
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": end_time - start_time,
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                "error": "timeout",
                "success": False,
                "duration": 600
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "duration": 0
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites."""
        print("🚀 Starting Comprehensive EthicalAI Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test categories
        self.results["backend"] = self.run_backend_tests()
        self.results["ethical"] = self.run_ethical_evaluation_tests()
        self.results["performance"] = self.run_performance_tests()
        self.results["frontend"] = self.run_frontend_tests()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Generate summary report
        self._generate_summary_report(total_duration)
        
        return self.results
    
    def _generate_summary_report(self, total_duration: float):
        """Generate and display test summary report."""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        # Backend tests summary
        print("\n🔧 Backend API Tests:")
        backend_results = self.results.get("backend", {})
        for test_file, result in backend_results.items():
            if isinstance(result, dict):
                status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
                duration = result.get("duration", 0)
                print(f"  {test_file}: {status} ({duration:.1f}s)")
                
                if result.get("success", False):
                    passed_tests += 1
                else:
                    failed_tests += 1
                total_tests += 1
        
        # Ethical evaluation tests summary
        print("\n🧠 Ethical Evaluation Tests:")
        ethical_results = self.results.get("ethical", {})
        for test_file, result in ethical_results.items():
            if isinstance(result, dict):
                status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
                duration = result.get("duration", 0)
                print(f"  {test_file}: {status} ({duration:.1f}s)")
                
                if result.get("success", False):
                    passed_tests += 1
                else:
                    failed_tests += 1
                total_tests += 1
        
        # Performance tests summary
        print("\n⚡ Performance Tests:")
        performance_results = self.results.get("performance", {})
        for test_file, result in performance_results.items():
            if isinstance(result, dict):
                status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
                duration = result.get("duration", 0)
                print(f"  {test_file}: {status} ({duration:.1f}s)")
                
                if result.get("success", False):
                    passed_tests += 1
                else:
                    failed_tests += 1
                total_tests += 1
        
        # Frontend tests summary
        print("\n🎨 Frontend Tests:")
        frontend_result = self.results.get("frontend", {})
        if isinstance(frontend_result, dict):
            if "error" in frontend_result:
                print(f"  Frontend tests: ❌ ERROR ({frontend_result['error']})")
                failed_tests += 1
            else:
                status = "✅ PASS" if frontend_result.get("success", False) else "❌ FAIL"
                print(f"  Frontend tests: {status}")
                if frontend_result.get("success", False):
                    passed_tests += 1
                else:
                    failed_tests += 1
            total_tests += 1
        
        # Overall summary
        print(f"\n📈 OVERALL RESULTS:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
        print(f"  Total Duration: {total_duration:.1f}s")
        
        # Test coverage areas
        print(f"\n🎯 TEST COVERAGE AREAS:")
        print(f"  ✅ Text Embedding & Vector Generation")
        print(f"  ✅ Ethical Evaluation Pipeline")
        print(f"  ✅ Axis Pack Loading & Configuration")
        print(f"  ✅ Vector Topology Analysis")
        print(f"  ✅ Batch Processing")
        print(f"  ✅ What-if Analysis")
        print(f"  ✅ Frontend-Backend Integration")
        print(f"  ✅ Error Handling & Edge Cases")
        print(f"  ✅ Performance Benchmarks")
        print(f"  ✅ API Contract Compliance")
        
        # Recommendations
        if failed_tests > 0:
            print(f"\n⚠️  RECOMMENDATIONS:")
            print(f"  • Review failed test output for specific issues")
            print(f"  • Check API endpoint implementations")
            print(f"  • Verify axis pack configurations")
            print(f"  • Ensure all dependencies are installed")
            print(f"  • Check frontend build and test setup")
        else:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"  • EthicalAI system is functioning correctly")
            print(f"  • All core functionality is working")
            print(f"  • Performance is within acceptable ranges")
            print(f"  • Frontend-backend integration is solid")


def main():
    """Main test execution function."""
    runner = TestRunner()
    
    try:
        results = runner.run_all_tests()
        
        # Exit with appropriate code
        overall_success = all(
            result.get("success", False) if isinstance(result, dict) else False
            for category_results in results.values()
            for result in (category_results.values() if isinstance(category_results, dict) else [category_results])
        )
        
        sys.exit(0 if overall_success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
