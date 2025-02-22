import asyncio
from playwright.async_api import async_playwright
from git_hala.download_base_img.base64_to_jpg import base64_to_jpg

async def get_base64_src(count):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  # Change to True if no UI needed
        page = await browser.new_page()
        await page.goto("https://electoralsearch.eci.gov.in/")  # Replace with your URL
        
        # Wait for the image to be loaded (if dynamically loaded)
        await page.wait_for_selector(".captcha-div img", timeout=5000)

        # Find the captcha container
        captcha_div = await page.query_selector(".captcha-div")
        
        if captcha_div:
            # Find the <img> tag inside
            img = await captcha_div.query_selector("img")

            if img:
                # Get the src attribute
                img_src = await img.get_attribute("src")
                
                if img_src:
                    print("Captcha Image Source:", img_src[:20])
                    base64_to_jpg(img_src, count)
                    count+=1
                else:
                    print("❌ Image src is empty!")
            else:
                print("❌ No image found inside .captcha-div")
        else:
            print("❌ Captcha div not found!")

        await browser.close()

count =102
while count <=100:
    asyncio.run(get_base64_src(count))


# 159929165