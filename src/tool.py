from typing import List

from langchain_core.tools import tool
from pptx import Presentation
from pydantic import BaseModel, Field


class PPTInput(BaseModel):
    filename: str = Field(description="輸出的檔案名稱，不需要包含副檔名")
    title: str = Field(description="簡報的主標題")
    bullet_points: List[str] = Field(description="簡報內容的重點列表")


@tool("generate_ppt", args_schema=PPTInput)
def generate_ppt(filename: str, title: str, bullet_points: List[str]) -> str:
    """用來生成策略總結 PPT 的工具。
    當使用者要求下載或產出簡報檔案時，請呼叫此工具。
    """
    try:
        prs = Presentation()
        # 建立標題頁
        slide_layout = prs.slide_layouts[1]  # Title and Content
        slide = prs.slides.add_slide(slide_layout)
        slide.shapes.title.text = title

        # 填入內容
        tf = slide.shapes.placeholders[1].text_frame
        for point in bullet_points:
            p = tf.add_paragraph()
            p.text = point

        # 儲存
        full_path = f"{filename}.pptx"
        prs.save(full_path)

        return f"成功生成檔案：{full_path}"
    except Exception as e:
        return f"生成失敗：{str(e)}"
