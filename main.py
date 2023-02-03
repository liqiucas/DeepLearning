from dataSet import loadDataSet

if __name__ == '__main__':
    # train_iter, test_iter = loadDataSet.load_data_mnist(batch_size=32, num_workers=4)
    #
    # train_iter2, test_iter2 = loadDataSet.load_data_fashion_mnist(batch_size=256, num_workers=4)
    # print(len(train_iter2), len(train_iter2))
    loadDataSet.testTime(dataset="load_data_fashion_mnist")
