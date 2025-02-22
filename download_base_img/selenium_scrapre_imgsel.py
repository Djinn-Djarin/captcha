from seleniumbase import Driver
from base64_jpg import base64_to_jpg
# Start the Undetected Chrome driver

# Open the target page
count =0
while count <=100:
    driver = Driver(uc=True, headless=False)  # Set headless=True to run in background
    driver.get("https://electoralsearch.eci.gov.in/")  # Replace with your target URL
    # Wait for the image inside `.captcha-div`
    driver.wait_for_element_visible(".captcha-div img", timeout=5)

    # Find the captcha image and get its `src`
    captcha_img = driver.find_element("css selector", ".captcha-div img")
    img_src = captcha_img.get_attribute("src")

    # Print the Base64-encoded image source
    if img_src:
        print("Captcha Image Source:", img_src[:10])
        base64_to_jpg(img_src, count)
        count+=1
    else:
        print("❌ Image src is empty!")
        continue

    driver.quit()
