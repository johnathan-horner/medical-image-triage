#!/usr/bin/env python3
"""
QA Test Script for Medical Image Triage App
Comprehensive testing for errors, functionality, and performance
"""

import asyncio
import json
from datetime import datetime
from playwright.async_api import async_playwright

async def test_medical_image_triage():
    """
    Comprehensive QA test for the medical image triage Streamlit app
    """
    async with async_playwright() as p:
        # Launch browser with extended timeout and debugging enabled
        browser = await p.chromium.launch(
            headless=False,  # Show browser for debugging
            args=['--disable-web-security', '--disable-features=VizDisplayCompositor']
        )

        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        )

        page = await context.new_page()

        # Capture console messages and errors
        console_logs = []
        network_errors = []

        def handle_console(msg):
            console_logs.append({
                'type': msg.type,
                'text': msg.text,
                'timestamp': datetime.now().isoformat()
            })
            print(f"Console {msg.type}: {msg.text}")

        def handle_request_failed(request):
            network_errors.append({
                'url': request.url,
                'method': request.method,
                'timestamp': datetime.now().isoformat()
            })
            print(f"Network Error: {request.method} {request.url}")

        page.on('console', handle_console)
        page.on('requestfailed', handle_request_failed)

        try:
            print("🔍 Starting QA Test for Medical Image Triage App")
            print(f"Target URL: https://medical-image-triage-horner.streamlit.app/")

            # Navigate to the app with extended timeout
            print("\n📍 Step 1: Navigating to application...")
            await page.goto(
                'https://medical-image-triage-horner.streamlit.app/',
                wait_until='networkidle',
                timeout=60000  # 60 second timeout
            )

            # Take initial screenshot
            await page.screenshot(path='/Users/johnathanhorner/medical-image-triage/qa_initial_load.png', full_page=True)
            print("✅ Initial screenshot saved: qa_initial_load.png")

            # Wait for Streamlit to fully load
            print("\n⏳ Step 2: Waiting for Streamlit to initialize...")
            try:
                # Look for common Streamlit elements
                await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)
                print("✅ Streamlit app container found")
            except Exception as e:
                print(f"⚠️  Streamlit container not found: {e}")

            # Check for error messages
            print("\n🔍 Step 3: Checking for error messages...")

            # Common error selectors
            error_selectors = [
                '[data-testid="stException"]',
                '.stException',
                '.error',
                '[data-testid="stAlert"]',
                '.stAlert',
                'div:has-text("error")',
                'div:has-text("Error")',
                'div:has-text("failed")',
                'div:has-text("Failed")',
                'div:has-text("not found")',
                'div:has-text("404")',
                'div:has-text("500")'
            ]

            errors_found = []
            for selector in error_selectors:
                try:
                    error_elements = await page.query_selector_all(selector)
                    for element in error_elements:
                        error_text = await element.inner_text()
                        if error_text.strip():
                            errors_found.append({
                                'selector': selector,
                                'text': error_text.strip()
                            })
                            print(f"🚨 Error found ({selector}): {error_text.strip()}")
                except Exception:
                    pass

            # Check page title and content
            print("\n📋 Step 4: Checking page content...")
            page_title = await page.title()
            print(f"Page Title: {page_title}")

            # Look for main content
            try:
                main_content = await page.query_selector('main')
                if main_content:
                    content_text = await main_content.inner_text()
                    print(f"Main content length: {len(content_text)} characters")
                    if len(content_text) < 100:
                        print("⚠️  Warning: Very little content found on page")
                else:
                    print("⚠️  No main content element found")
            except Exception as e:
                print(f"❌ Error checking main content: {e}")

            # Check for specific Streamlit elements
            print("\n🎯 Step 5: Checking Streamlit-specific elements...")
            streamlit_elements = {
                'sidebar': '[data-testid="stSidebar"]',
                'header': '[data-testid="stHeader"]',
                'main_content': '[data-testid="stMain"]',
                'file_uploader': '[data-testid="stFileUploader"]'
            }

            for element_name, selector in streamlit_elements.items():
                try:
                    element = await page.query_selector(selector)
                    if element:
                        print(f"✅ {element_name} found")
                    else:
                        print(f"⚠️  {element_name} not found")
                except Exception as e:
                    print(f"❌ Error checking {element_name}: {e}")

            # Take screenshot of current state
            await page.screenshot(path='/Users/johnathanhorner/medical-image-triage/qa_after_checks.png', full_page=True)
            print("✅ Post-check screenshot saved: qa_after_checks.png")

            # Test file upload functionality if available
            print("\n📤 Step 6: Testing file upload functionality...")
            try:
                file_uploader = await page.query_selector('input[type="file"]')
                if file_uploader:
                    print("✅ File upload input found")
                    # Take screenshot of file upload area
                    await page.screenshot(path='/Users/johnathanhorner/medical-image-triage/qa_file_upload.png')
                else:
                    print("⚠️  No file upload input found")
            except Exception as e:
                print(f"❌ Error testing file upload: {e}")

            # Wait a bit more to catch any delayed errors
            print("\n⏳ Step 7: Waiting for delayed errors...")
            await page.wait_for_timeout(5000)

            # Final screenshot
            await page.screenshot(path='/Users/johnathanhorner/medical-image-triage/qa_final.png', full_page=True)
            print("✅ Final screenshot saved: qa_final.png")

            # Generate test report
            print("\n📊 GENERATING QA REPORT")
            print("=" * 50)

            # Page Status
            print(f"✅ Page loaded successfully: {page.url}")
            print(f"📄 Page title: {page_title}")

            # Console Logs Summary
            print(f"\n📝 Console Messages: {len(console_logs)} total")
            if console_logs:
                error_logs = [log for log in console_logs if log['type'] == 'error']
                warning_logs = [log for log in console_logs if log['type'] == 'warning']

                print(f"  - Errors: {len(error_logs)}")
                print(f"  - Warnings: {len(warning_logs)}")
                print(f"  - Other: {len(console_logs) - len(error_logs) - len(warning_logs)}")

                if error_logs:
                    print("\n🚨 CONSOLE ERRORS:")
                    for log in error_logs[-5:]:  # Show last 5 errors
                        print(f"  - {log['text']}")

            # Network Errors
            print(f"\n🌐 Network Errors: {len(network_errors)}")
            if network_errors:
                print("🚨 NETWORK FAILURES:")
                for error in network_errors[-5:]:  # Show last 5 network errors
                    print(f"  - {error['method']} {error['url']}")

            # UI Errors
            print(f"\n🎨 UI Errors Found: {len(errors_found)}")
            if errors_found:
                print("🚨 UI ERROR MESSAGES:")
                for error in errors_found:
                    print(f"  - {error['text']}")

            # Quality Assessment
            print(f"\n🎯 QUALITY ASSESSMENT")
            total_issues = len(console_logs) + len(network_errors) + len(errors_found)

            if total_issues == 0:
                print("✅ EXCELLENT: No issues detected")
            elif total_issues <= 5:
                print("⚠️  GOOD: Minor issues detected")
            elif total_issues <= 15:
                print("🚨 FAIR: Multiple issues detected - needs attention")
            else:
                print("❌ POOR: Serious issues detected - immediate action required")

            # Save detailed log
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'url': page.url,
                'title': page_title,
                'console_logs': console_logs,
                'network_errors': network_errors,
                'ui_errors': errors_found,
                'total_issues': total_issues
            }

            with open('/Users/johnathanhorner/medical-image-triage/qa_report.json', 'w') as f:
                json.dump(report_data, f, indent=2)

            print(f"\n💾 Detailed report saved: qa_report.json")
            print("📸 Screenshots saved: qa_initial_load.png, qa_after_checks.png, qa_final.png")

        except Exception as e:
            print(f"❌ CRITICAL ERROR during testing: {e}")
            # Take error screenshot
            await page.screenshot(path='/Users/johnathanhorner/medical-image-triage/qa_error.png', full_page=True)

        finally:
            await browser.close()
            print("\n🔚 QA Test Complete")

if __name__ == "__main__":
    asyncio.run(test_medical_image_triage())