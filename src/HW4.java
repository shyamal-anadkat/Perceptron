import java.util.*;
import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;

///////////////////////////////
// Shyamal H Anadkat        ///                     
// HW4 - Perceptron         ///
// CS540 (Shavlik)          ///                                                        
///////////////////////////////

public class HW4
{

	/***********************VARIABLES********************************/
	public final static double ALPHA_RATE = 0.1;
	static ListOfExamples tuneSet = new ListOfExamples(); 
	static ListOfExamples testSet = new ListOfExamples();
	static ListOfExamples trainSet = new ListOfExamples();
	public static List<String> firstVals = new ArrayList<String>();
	public static List<String> secondVals = new ArrayList<String>();
	static int numInputs = 0;
	static int maxEpoche = 1000;
	static final boolean DEBUG = false;
	/*****************************************************************/

	//RANDOM INSTANCE//
	public static Random randomInstance = new Random(540); // Change 540 to another integer to get different results.
	public static double random() { // Call this when you use random numbers to permute your list of training examples. 
		// Recall one can permute a list of examples by
		//    (a) assigning a random number to each example,
		//    (b) sort based on these assigned numbers, and
		//    (c) remove the random numbers (actually, could just keep them in your Example class, but don't use these as a feature during learning!)
		return randomInstance.nextDouble();
	}


	public static void main(String[] args)
	{   
		if (args.length != 3) {
			System.err.println("Please supply 3 filenames on the " +
					"command line: java ScannerSample" + 
					" <filename>");
			System.exit(1);
		}

		//initialize and read tune and testset examples for red-wine-quality. 
		System.out.println("WELCOME TO CS540 PERCEPTRON - SHYAMAL ANADKAT");
		System.out.println("---------------------------------------------");
		trainSet.ReadInExamplesFromFile(args[0]);
		tuneSet.ReadInExamplesFromFile(args[1]);
		testSet.ReadInExamplesFromFile(args[2]);

		numInputs = trainSet.getNumberOfFeatures() + 1;
		int maxEpochForTune= 50;
		Double maxTune= 0.0,maxTest=0.0;

		//initialize vector for learning model, this will store weights
		Vector<Double> learnedModel = new Vector<Double>();
		initWeightsInPerceptron(learnedModel, 0.0, numInputs); //initialize all with 0

		Vector<Double> bestModel = null;
		int epoche = 0;int step = 50;

		while(epoche != maxEpoche) {
			epoche = epoche+50;
			trainPerceptron(trainSet, learnedModel, step);
			System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++");
			System.out.println("Epoche: "+epoche);
			System.out.println("Weights: "+learnedModel);
			Double currTest = testPerceptron(learnedModel, testSet, trainSet);
			Double currTrain = testPerceptron(learnedModel, trainSet, trainSet);
			Double currTune = testPerceptron(learnedModel, tuneSet, trainSet);
			//print percent accuracies rounded to 5 decimal places 
			System.out.println("Accuracy for train set: "+Utilities.round(currTrain, 5)+"%");
			System.out.println("Accuracy for tune set: "+Utilities.round(currTune, 5)+"%");
			System.out.println("Accuracy for test set: "+Utilities.round(currTest, 5)+"%");
			//getting perceptron with best tune 
			if(currTune > maxTune) {
				maxTune = currTune; 
				maxEpochForTune = epoche;
				maxTest = currTest;
				bestModel = new Vector(learnedModel); //update best model/perceptron here
			};
		}
		System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++");
		//print out weights and features for best perceptron
		for(int i = 0; i < trainSet.getNumberOfFeatures(); i ++) {
			System.out.println(i+1+") Wgt = "+Utilities.round(bestModel.get(i), 3)+" | "+trainSet.getFeatures()[i].getName());
		}
		System.out.println("Threshold: "+bestModel.get(bestModel.size()-1));
		System.out.println("////////////////////////////////////////////////////////");
		System.out.println("Max Epoche for Tune:"+maxEpochForTune+" at "+maxTune+"%");
		System.out.println("Max Test Accuracy here: "+maxTest+"%");

		//outdated debug
		if(DEBUG) {
			System.out.println("Number of inputs to perceptron: "+numInputs);
			System.out.println("Size of learned model: "+ learnedModel.size());
			tuneSet.DescribeDataset();
			trainSet.DescribeDataset();
			testSet.DescribeDataset();
		}

		System.out.println("---------------------------------");
		Utilities.waitHere("Hit <enter> when ready to exit.");
	}

	/**
	 * Perceptron learning algorithm 
	 * @param loe
	 * @param model
	 */
	public static void trainPerceptron(ListOfExamples loe, Vector<Double> model, int epoche){
		//run it for num epoches (steps of 50, could vary)
		for(int y = 0; y < epoche; y ++){
			loe = permuteOrder(loe);  //permuting order of the list 
			for(int ex = 0; ex < loe.size(); ex++) {
				ArrayList<Double> inputs = new ArrayList<Double>();
				Example curr = loe.get(ex); //curr example

				int y_out = curr.getLabel().equals(loe.getOutputLabels().getFirstValue()) ? 0:1;
				for(int i = 0; i < curr.size(); i++) { //populate inputs with feature vals
					inputs.add(firstVals.contains(curr.get(i)) ? 0.0:1.0);
				}
				Double weightedInput = getWeightedSum(model, inputs);
				int hyp_out = weightedInput >= model.get(model.size()-1) ? 1:0;
				//System.out.println(hyp_out+"-"+y_out);
				if(y_out != hyp_out) {
					//update the weights and learn 
					for(int i = 0; i < model.size() - 1 ; i++) {
						Double deltaW = ALPHA_RATE * (double)(y_out - hyp_out) * inputs.get(i) ; 
						model.set(i, (double)(model.get(i) + deltaW));
					}
					//update threshold which is the last value in learnedModel
					Double thresholdDelta = ALPHA_RATE * (double)(y_out - hyp_out) * -1 ;
					model.set(model.size()-1, (model.get(model.size()-1)+thresholdDelta));
				}
			}
		}
	}

	/**
	 * Testing perceptron model against test examples
	 * @param model
	 * @param test_loe
	 * @param train
	 */
	public static Double testPerceptron(Vector<Double> model, ListOfExamples test_loe, ListOfExamples train) {
		int positives = 0; 
		for(int x = 0; x < test_loe.size(); x++) {
			Example curr = test_loe.get(x);
			int label = curr.getLabel().equals(train.getOutputLabels().getFirstValue()) ? 0:1;
			if(classifyExample(model,test_loe.get(x)) == label) {
				positives++;
			}
		}
		return (double)positives/test_loe.size()*100;
	}

	/**
	 * Classifying example using the learned model
	 * @param model
	 * @param test
	 * @return
	 */
	private static int classifyExample(Vector<Double> model, Example test){
		ArrayList<Double> inputs = new ArrayList<Double>(); 
		for(int i = 0; i < test.size(); i++) {
			inputs.add(firstVals.contains(test.get(i)) ? 0.0:1.0);
		}
		Double weightedInput = getWeightedSum(model, inputs);
		return weightedInput >= model.get(model.size()-1) ? 1:0; 
	}


	/**
	 * Permuting example order using given random instance seed
	 * @param in
	 * @return
	 */
	public static ListOfExamples permuteOrder(ListOfExamples in) {
		Collections.shuffle(in, randomInstance);
		return in; 
	}

	/**
	 * Initializing weights with given val 
	 * @param in
	 * @param val
	 */
	private static void initWeightsInPerceptron(Vector<Double> in, Double val, int size) {
		for(int i = 0; i < size; i ++) {
			in.add(val);
		}
	}

	/**
	 * Calculate the weighted sum 
	 * @param weights
	 * @param inp
	 * @return
	 */
	public static Double getWeightedSum(Vector<Double> weights, ArrayList<Double> inp) {
		Double retVal = 0.0; 
		for(int i = 0;  i < inp.size()-1; i++) {
			retVal = retVal + (inp.get(i) * weights.get(i));
		}
		// for the -1 unit / BIAS
		retVal += (inp.get(inp.size()-1) * weights.get(weights.size()-1));
		retVal += (-1 * weights.get(weights.size() - 1));
		return retVal; 
	}

}

// This class, an extension of ArrayList, holds an individual example.
// The new method PrintFeatures() can be used to
// display the contents of the example. 
// The items in the ArrayList are the feature values.
class Example extends ArrayList<String>
{
	// The name of this example.
	private String name;  

	// The output label of this example.
	private String label;

	// The data set in which this is one example.
	private ListOfExamples parent;  

	// Constructor which stores the dataset which the example belongs to.
	public Example(ListOfExamples parent) {
		this.parent = parent;
	}

	// Print out this example in human-readable form.
	public void PrintFeatures()
	{
		System.out.print("Example " + name + ",  label = " + label + "\n");
		for (int i = 0; i < parent.getNumberOfFeatures(); i++)
		{
			System.out.print("     " + parent.getFeatureName(i)
			+ " = " +  this.get(i) + "\n");
		}
	}

	// Adds a feature value to the example.
	public void addFeatureValue(String value) {
		this.add(value);
	}

	// Accessor methods.
	public String getName() {
		return name;
	}

	public String getLabel() {
		return label;
	}

	// Mutator methods.
	public void setName(String name) {
		this.name = name;
	}

	public void setLabel(String label) {
		this.label = label;
	}
}

/* This class holds all of our examples from one dataset
   (train OR test, not BOTH).  It extends the ArrayList class.
   Be sure you're not confused.  We're using TWO types of ArrayLists.  
   An Example is an ArrayList of feature values, while a ListOfExamples is 
   an ArrayList of examples. Also, there is one ListOfExamples for the 
   TRAINING SET and one for the TESTING SET. 
 */
class ListOfExamples extends ArrayList<Example>
{
	// The name of the dataset.
	private String nameOfDataset = "";

	// The number of features per example in the dataset.
	private int numFeatures = -1;

	// An array of the parsed features in the data.
	private BinaryFeature[] features;

	// A binary feature representing the output label of the dataset.
	private BinaryFeature outputLabel;

	// The number of examples in the dataset.
	private int numExamples = -1;

	public ListOfExamples() {} 

	// Print out a high-level description of the dataset including its features.
	public void DescribeDataset()
	{
		System.out.println("Dataset '" + nameOfDataset + "' contains "
				+ numExamples + " examples, each with "
				+ numFeatures + " features.");
		System.out.println("Valid category labels: "
				+ outputLabel.getFirstValue() + ", "
				+ outputLabel.getSecondValue());
		System.out.println("The feature names (with their possible values) are:");
		for (int i = 0; i < numFeatures; i++)
		{
			BinaryFeature f = features[i];
			System.out.println("   " + f.getName() + " (" + f.getFirstValue() +
					" or " + f.getSecondValue() + ")");
		}
	}

	public int getNumFeatures(){
		return this.numFeatures;
	}
	public BinaryFeature[] getFeatures() {
		return this.features;
	}

	public void setFeatures(BinaryFeature[] a) {
		this.features = a; 
	}
	public void setNumFeatures(int num){
		this.numFeatures = num;
	}
	public int getNumExamples() {
		return this.numExamples;
	}
	public void setNumExamples(int exs){
		this.numExamples = exs; 
	}
	public void setOutputLabel(BinaryFeature label) {
		this.outputLabel = label;
	}


	/**
	 * Gets count of a particular label from set of examples 
	 * @param label
	 * @return
	 */
	public int getLabelCount(String label) {
		int count = 0; 
		for(int i = 0; i < size(); i++) {
			if(this.get(i).getLabel().equalsIgnoreCase(label)) {
				count++;
			}
		}
		return count;
	}

	/**
	 * Get output label 
	 * @return
	 */
	public BinaryFeature getOutputLabels() {
		return this.outputLabel;
	}

	public void filterExample(Example ex){
		this.remove(ex);
	}

	// Print out ALL the examples.
	public void PrintAllExamples()
	{
		System.out.println("List of Examples\n================");
		for (int i = 0; i < size(); i++)
		{
			Example thisExample = this.get(i);  
			thisExample.PrintFeatures();
		}
	}

	// Print out the SPECIFIED example.
	public void PrintThisExample(int i)
	{
		Example thisExample = this.get(i); 
		thisExample.PrintFeatures();
	}

	// Returns the number of features in the data.
	public int getNumberOfFeatures() {
		return numFeatures;
	}

	// Returns the name of the ith feature.
	public String getFeatureName(int i) {
		return features[i].getName();
	}

	public BinaryFeature[] getfeatures() {
		return features;
	}

	// Takes the name of an input file and attempts to open it for parsing.
	// If it is successful, it reads the dataset into its internal structures.
	// Returns true if the read was successful.
	public boolean ReadInExamplesFromFile(String dataFile) {
		nameOfDataset = dataFile;
		// Try creating a scanner to read the input file.
		Scanner fileScanner = null;
		try {
			fileScanner = new Scanner(new File(dataFile));
		} catch(FileNotFoundException e) {
			return false;
		}
		// If the file was successfully opened, read the file
		this.parse(fileScanner);
		return true;
	}

	/**
	 * Does the actual parsing work. We assume that the file is in proper format.
	 *
	 * @param fileScanner a Scanner which has been successfully opened to read
	 * the dataset file
	 */
	public void parse(Scanner fileScanner) {
		// Read the number of features per example.
		numFeatures = Integer.parseInt(parseSingleToken(fileScanner));

		// Parse the features from the file.
		parseFeatures(fileScanner);

		// Read the two possible output label values.
		String labelName = "output";
		String firstValue = parseSingleToken(fileScanner);
		String secondValue = parseSingleToken(fileScanner);
		outputLabel = new BinaryFeature(labelName, firstValue, secondValue);

		// Read the number of examples from the file.
		numExamples = Integer.parseInt(parseSingleToken(fileScanner));

		parseExamples(fileScanner);
	}

	/**
	 * Returns the first token encountered on a significant line in the file.
	 *
	 * @param fileScanner a Scanner used to read the file.
	 */
	private String parseSingleToken(Scanner fileScanner) {
		String line = findSignificantLine(fileScanner);
		// Once we find a significant line, parse the first token on the
		// line and return it.
		Scanner lineScanner = new Scanner(line);
		return lineScanner.next();
	}

	/**
	 * Reads in the feature metadata from the file.
	 * 
	 * @param fileScanner a Scanner used to read the file.
	 */
	private void parseFeatures(Scanner fileScanner) {
		// Initialize the array of features to fill.
		features = new BinaryFeature[numFeatures];
		for(int i = 0; i < numFeatures; i++) {
			String line = findSignificantLine(fileScanner);
			// Once we find a significant line, read the feature description
			// from it.
			Scanner lineScanner = new Scanner(line);
			String name = lineScanner.next();
			String dash = lineScanner.next();  // Skip the dash in the file.
			String firstValue = lineScanner.next();
			HW4.firstVals.add(firstValue);
			String secondValue = lineScanner.next();
			HW4.secondVals.add(secondValue);
			features[i] = new BinaryFeature(name, firstValue, secondValue);
		}
	}

	/**
	 * Parse example from Scanner 
	 * @param fileScanner
	 */
	private void parseExamples(Scanner fileScanner) {
		// Parse the expected number of examples.
		for(int i = 0; i < numExamples; i++) {
			String line = findSignificantLine(fileScanner);
			Scanner lineScanner = new Scanner(line);

			// Parse a new example from the file.
			Example ex = new Example(this);

			String name = lineScanner.next();
			ex.setName(name);

			String label = lineScanner.next();
			ex.setLabel(label);

			// Iterate through the features and increment the count for any feature
			// that has the first possible value.
			for(int j = 0; j < numFeatures; j++) {
				String feature = lineScanner.next();
				ex.addFeatureValue(feature);
			}
			// Add this example to the list.
			this.add(ex);
		}
	}

	/**
	 * Returns the next line in the file which is significant (i.e. is not
	 * all whitespace or a comment.
	 *
	 * @param fileScanner a Scanner used to read the file
	 */
	private String findSignificantLine(Scanner fileScanner) {
		// Keep scanning lines until we find a significant one.
		while(fileScanner.hasNextLine()) {
			String line = fileScanner.nextLine().trim();
			if (isLineSignificant(line)) {
				return line;
			}
		}
		// If the file is in proper format, this should never happen.
		System.err.println("Unexpected problem in findSignificantLine.");
		return null;
	}

	/**
	 * Returns whether the given line is significant (i.e., not blank or a
	 * comment). The line should be trimmed before calling this.
	 *
	 * @param line the line to check
	 */
	private boolean isLineSignificant(String line) {
		// Blank lines are not significant.
		if(line.length() == 0) {
			return false;
		}
		// Lines which have consecutive forward slashes as their first two
		// characters are comments and are not significant.
		if(line.length() > 2 && line.substring(0,2).equals("//")) {
			return false;
		}
		return true;
	}
}

/**
 * Represents a single binary feature with two String values.
 */
class BinaryFeature {
	private String name;
	private String firstValue;
	private String secondValue;

	public BinaryFeature(String name, String first, String second) {
		this.name = name;
		firstValue = first;
		secondValue = second;
	}

	public String getName() {
		return name;
	}

	public String getFirstValue() {
		return firstValue;
	}

	public String getSecondValue() {
		return secondValue;
	}
}

/**
 * 
 * @author SAnadkat
 *
 */
class Utilities
{
	// This method can be used to wait until you're ready to proceed.
	public static void waitHere(String msg)
	{
		System.out.print("\n" + msg);
		try { System.in.read(); }
		catch(Exception e) {} // Ignore any errors while reading.
	}
	public static double round(double value, int places) {
		if (places < 0) throw new IllegalArgumentException();

		BigDecimal bd = new BigDecimal(value);
		bd = bd.setScale(places, RoundingMode.HALF_UP);
		return bd.doubleValue();
	}
}
