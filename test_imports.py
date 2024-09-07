from src.components.data_transformation import DataTransformation

def test_import():
    try:
        # Instantiate DataTransformation to see if the import works
        data_transformation = DataTransformation()
        print("Import successful!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_import()
