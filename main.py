from src.graph import app
from src.logger import logger


def main():
    print("=== 開始測試 Agent Graph ===")

    # 1. 定義初始狀態 (Initial State)
    initial_state = {
        "messages": [],
        "cur_step": "start",
        "situation_summary": "",
        "reflection_passed": False,
    }

    # 2. 執行 Graph
    # invoke 會同步執行整個流程直到 END
    try:
        result = app.invoke(initial_state)
        print("=== 測試完成 ===")
        print("最終狀態:", result)
    except Exception as e:
        logger.error(f"執行發生錯誤: {e}")
        print(f"發生錯誤: {e}")


if __name__ == "__main__":
    main()
