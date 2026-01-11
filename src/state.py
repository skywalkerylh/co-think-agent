from typing import Any, List, Optional

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated


class ProblemExtraction(BaseModel):
    job_title: Optional[str] = Field(description="用戶的職位，若無則留空")
    pain_point: Optional[str] = Field(description="用戶提到的問題痛點，若無則留空")
    goal: Optional[str] = Field(description="用戶想達成的目標，若無則留空")


class EvaluationDimensions(BaseModel):
    pain_point_score: int = Field(..., description="痛點描述的具體程度 (0-30)")
    goal_metric_score: int = Field(..., description="目標與指標的清晰度 (0-40)")
    box_trap_score: int = Field(..., description="是否跳脫『手段當目的』的陷阱 (0-30)")


class ProblemEvaluation(BaseModel):
    score: int = Field(..., description="總分 (0-100)")
    dimensions: EvaluationDimensions
    is_passing: bool = Field(..., description="是否通過門檻")
    critique: str = Field(..., description="犀利的評語")
    advice: str = Field(..., description="給用戶的引導建議")
    missing_fields: list[str] = Field(..., description="缺少的關鍵資訊欄位")


class CrossSiloOutput(BaseModel):
    result: Optional[str] = Field(..., description="跨部門視角的見解")
    score: int = Field(..., description="策略完整度分數 (0-100)")


class State(BaseModel):
    messages: Annotated[List[Any], add_messages]
   
    problem_profile: dict = Field(
        default_factory=lambda: {
            "pain_point": None,  
            "goal": None, 
        }
    )

    reflection_result: dict = Field(
        default_factory=lambda: {
            "is_complete": False,
            "missing_fields": [],  
        }
    )
    evaluation_result: dict = Field(
        default_factory=lambda: {
            "score": 0,
            "critique": "",
            "advice": "",
            "missing_fields": [],
        }
    )
    cross_silo_output: dict = Field(
        default_factory=lambda: {
            "result": "",
            "score": 0,
        }
    )
    job_title: Optional[str] = None
    is_passing_evaluation: bool = False
    node_status: str = "example"
    last_stage: str = "" # 用來記錄最後執行的節點名稱
    count_node_file_export: int = 0
