# 關鍵的 Loop 邏輯
def check_quality(state):
    if state["reflection_passed"]:
        return "cross_silo"
    else:
        return "situation"
