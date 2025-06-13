from models.predict_model import predict

if __name__ == '__main__':
    # Change filenames if needed!
    test_file = './data/Hackathon_bureau_data_50000.csv'  # Or hidden/test csv provided at the hackathon
    output_file = './output/output_sample.csv'
    predict(test_file, output_file)
