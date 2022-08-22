from utils import DataDetails
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class SimplePipeline:
    def __init__(self):
        self.frame = None
        # specify that each value should start out as
        # None when the class is instantiated.
        self.X_train, self.X_test, self.y_train, self.Y_test = None, None, None, None
        self.model = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load the dataset and perform train test split."""
        # load the data
        dataset = pd.read_csv(DataDetails.datafile)
        
        # remove units ' (cm)' from variable names
        self.feature_names = dataset.columns 
        self.frame = dataset
        self.frame['target'] = dataset['status']
        
        # we divide the data set using the train_test_split function from sklearn, 
        # which takes as parameters, the dataframe with the predictor variables, 
        # then the target, then the percentage of data to assign to the test set, 
        # and finally the random_state to ensure reproducibility.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.frame[self.feature_names], self.frame.target, test_size=0.65, random_state=42)
        
    def train(self, algorithm=LogisticRegression):
        
        # we set up a LogisticRegression classifier with default parameters
        self.model = algorithm(solver='lbfgs', multi_class='auto')
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self, input_data):
        return self.model.predict(input_data)
        
    def get_accuracy(self):
        
        # use our X_test and y_test values generated when we used
        # `train_test_split` to test accuracy.
        # score is a method on the Logisitic Regression that 
        # returns the accuracy by default, but can be changed to other metrics, see: 
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.score
        return self.model.score(X=self.X_test, y=self.y_test)
    
    def run_pipeline(self):
        """Helper method to run multiple pipeline methods with one call."""
        self.load_dataset()
        self.train()