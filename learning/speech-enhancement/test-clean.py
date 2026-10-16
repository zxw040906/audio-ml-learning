
from pathlib import Path


def main():
    # 获取当前 Python 文件所在目录
    base_dir = Path(__file__).resolve().parent

    # 练习 1：填写独立测试集目录
    test_dir = base_dir / "data" / "LibriSpeech" / "test-clean"

    # 检查目录是否存在
    if not test_dir.exists():
        print("测试集目录不存在:", test_dir)
        return

    # 练习 2：递归查找所有 FLAC 音频
    test_files = sorted(test_dir.rglob("*.flac"))

    # 输出检查结果
    print("测试集目录:", test_dir)
    print("测试集音频数量:", len(test_files))

    if len(test_files) > 0:
        print("第一条测试音频:", test_files[0])


if __name__ == "__main__":
    main()
