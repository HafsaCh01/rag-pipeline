import sys
def check_imports():
    print("=== checking imports ===")
    mods ={}
    try:
        import datasets
        mods["datasets"] = datasets.__version__
        import pandas as pd
        mods["pandas"] = pd.__version__
        import numpy as np
        mods["numpy"] = np.__version__
        import langdetect
        mods["langdetect"] = "OK"
        import ftfy
        mods["ftfy"] = "OK"
        import bs4
        mods["beautifulsoup4"] = bs4.__version__
        import tqdm
        mods["tqdm"] = tqdm.__version__
        import pytest
        mods["pytest"] = pytest.__version__
    except ImportError as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    for name, version in mods.items():
        print(f" {name}: {version}")
        return True
    
def check_basic_functionality():
    print("\n=== Checking basic functionality  ===")

    import pandas as pd
    df = pd.DataFrame({"text": ["hello", None, "world"]})
    assert df["text"].isnull().sum() == 1
    print("pandas DataFrame + null detection works")

    import numpy as np
    arr = np.array([1, 2, 3])
    assert arr.mean() == 2.0
    print(" numpy array ops work")

    import ftfy
    fixed = ftfy.fix_text("Ã¢â‚¬â„¢")
    assert isinstance(fixed, str)
    print("ftfy.fix_text runs")

    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    lang = detect("This is a sentence in English.")
    assert lang == "en"
    print(f" langdetect works (detected: {lang})")

    from bs4 import BeautifulSoup
    soup = BeautifulSoup("<p>Hello <b>world</b></p>", "lxml")
    assert soup.get_text() == "Hello world"
    print("BeautifulSoup + lxml parser works")

    import datasets
    ds = datasets.Dataset.from_dict({"text": ["a", "b", "c"]})
    assert len(ds) == 3
    print("HuggingFace datasets in-memory Dataset works")

    print("\nAll checks passed.")


if __name__ == "__main__":
    check_imports()
    check_basic_functionality()