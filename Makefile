.PHONY: test verify

# 规范自检：语法 + stub + C 编译 + 关键回归（校准路径/init_runtime/奇数核）
test: verify

verify:
	python3 scripts/verify.py
