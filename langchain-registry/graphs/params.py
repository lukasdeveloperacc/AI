from dataclasses import dataclass

import requests, re
import xml.etree.ElementTree as ET


@dataclass
class DOME79:
    url: str = "https://www.79dome.com/Api/ProductSelect_Api.php"
    id: str = "lukasmarketacc"
    api_key: str = "bHVrYXNtYXJrZXRhY2NfMTMzLTI2"

    @staticmethod
    def get_detail_image_page_url(product_id: str) -> list[str]:
        url = f"{DOME79.url}?id={DOME79.id}&apiKey={DOME79.api_key}&goodsno={product_id}"

        try:
            with requests.get(url) as response:
                response.raise_for_status()
                xml_data = response.text

            root = ET.fromstring(xml_data)
            detailed_source = root.find(".//detailed_source").text
            img_srcs = re.findall(r'<img src="(.*?)"', detailed_source, re.IGNORECASE)

            if len(img_srcs) < 1:
                raise ValueError(f"img_srcs is empty : {img_srcs}")

            return img_srcs

        except Exception as e:
            raise Exception(f"Erorr : {e}")


@dataclass
class NaverAPI:
    url: str = "https://api.naver.com/keywordstool"
    api_key: str = "010000000004be245f8c13172c2080aa7a15c2371fecd403fac7880664e0f4270cffef9465"
    secret_key: str = "AQAAAAAEviRfjBMXLCCAqnoVwjcfFJR8HEqotFMi7371Mu0HzA=="
    customer_id: str = "3374497"
