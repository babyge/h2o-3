package hex.gam;

import Jama.Matrix;
import Jama.QRDecomposition;
import hex.gam.GamSplines.ThinPlateDistanceWithKnots;
import org.junit.Test;
import org.junit.runner.RunWith;
import water.DKV;
import water.Scope;
import water.TestUtil;
import water.fvec.Frame;
import water.runner.CloudSize;
import water.runner.H2ORunner;
import water.util.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static hex.gam.GAMModel.GAMParameters;
import static hex.gam.GAMModel.GAMModelOutput;
import static hex.gam.GamSplines.ThinPlateRegressionUtils.*;
import static hex.genmodel.algos.gam.GamUtilsThinPlateRegression.calculateDistance;
import static hex.util.LinearAlgebraUtils.generateOrthogonalComplement;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static water.util.ArrayUtils.*;

@RunWith(H2ORunner.class)
@CloudSize(1)
public class GamThinPlateRegressionBasicTest extends TestUtil {
  public static final double MAGEPS = 1e-10;
  
  // test correct generation of orthogonal matrix zCS
  @Test
  public void testGenOrthComplement() {
    double[][] matT = new double[][]{{1.0, -19.8, -1.99}, {1.0, -0.97, -0.98}, {1.0, 0.031, 0.03}, {1.0, 0.06, 0.05},
            {1.0, 1.0, 1.01}, {1.0, 1.98, 1.99}};
    Matrix starTMat = new Matrix(matT);        // generate Zcs as in 3.3
    QRDecomposition starTMat_qr = new QRDecomposition(starTMat);
    double[][] qMatT = ArrayUtils.transpose(starTMat_qr.getQ().getArray()); // contains orthogonal basis transpose
    double[][] zCST = generateOrthogonalComplement(matT, 3, 12345);
    // check zCS: zCS should be orthogonal to qMat and to each other.  They also should have unit magnitude.
    int numQ = qMatT.length;
    int numZ = zCST.length;
    // check zCS orthogonal to qMatT
    for (int index = 0; index < numQ; index++)
      for (int indexB = 0; indexB < numZ; indexB++)
        assertTrue(Math.abs(innerProduct(qMatT[index], zCST[indexB])) < MAGEPS);
    // check zCS is orthogonal and have unit magnitude
    for (int index = 0; index < numZ; index++) {
      for (int indexB = 0; indexB < numZ; indexB++) {
        if (indexB == index)
          assertTrue(Math.abs(innerProduct(zCST[index], zCST[indexB]) - 1) < MAGEPS);
        else
          assertTrue(Math.abs(innerProduct(zCST[index], zCST[indexB])) < MAGEPS);
      }
    }
  }

  // test with multinomial with only thin plate regressioon smoothers with one predictors only
  @Test
  public void testTP1D() {
    Scope.enter();
    try {
      Frame train = Scope.track(parse_test_file("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv"));
      String[] ignoredCols = new String[]{"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
      String[][] gamCols = new String[][]{{"C6"}, {"C7"}, {"C8"}};
      train.replace((10), train.vec(10).toCategoricalVec()).remove();
      DKV.put(train);
      GAMParameters params = new GAMParameters();
      int k = 10;
      params._response_column = "C11";
      params._ignored_columns = ignoredCols;
      params._num_knots = new int[]{k, k, k};
      params._gam_columns = gamCols;
      params._bs = new int[]{1, 1, 1};
      params._scale = new double[]{10, 10, 10};
      params._train = train._key;
      params._savePenaltyMat = true;
      params._lambda_search = true;
      GAMModel gam = new GAM(params).trainModel().get();
      // check starT is of size k x M
      assertTrue((gam._output._starT[0].length == k) && (gam._output._starT[0][0].length == params._M[0]));
      // check penalty_CS is size k x k
      assertTrue((gam._output._penaltyMatCS[0].length == (k-params._M[0])) &&
              (gam._output._penaltyMatCS[0][0].length == (k-params._M[0])));
      Scope.track_generic(gam);
    } finally {
      Scope.exit();
    }
  }

  // test with Gaussian with only thin plate regression smoothers with two predictors.  
  @Test
  public void testTP2D() {
    Scope.enter();
    try {
      Frame train = Scope.track(parse_test_file("smalldata/glm_test/gaussian_20cols_10000Rows.csv"));
      String[] ignoredCols = new String[]{"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", 
              "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20"};
      String[][] gamCols = new String[][]{{"C11", "C12"}, {"C13", "C14"}};
      GAMParameters params = new GAMParameters();
      int k = 6;
      params._response_column = "C21";
      params._ignored_columns = ignoredCols;
      params._num_knots = new int[]{k, k};
      params._gam_columns = gamCols;
      params._bs = new int[]{1, 1};
      params._scale = new double[]{0.01, 0.01};
      params._train = train._key;
      params._savePenaltyMat = true;
      params._lambda = new double[]{0.01};
      //params._lambda_search = true;
      GAMModel gam = new GAM(params).trainModel().get();
      Scope.track_generic(gam);
      // check starT is of size k x M
      assertTrue((gam._output._starT[0].length == k) && (gam._output._starT[0][0].length == params._M[0]));
      // check penalty_CS is size k x k
      assertTrue((gam._output._penaltyMatCS[0].length == (k-params._M[0])) &&
              (gam._output._penaltyMatCS[0][0].length == (k-params._M[0])));
    } finally {
      Scope.exit();
    }
  }
  
  // test with binomial for thin plate regression smoothers with three predictors and check that the polynomials
  // are generated correctly by checking starT values
  @Test
  public void testTP3DStarT() {
    Scope.enter();
    try {
      Frame train = Scope.track(parse_test_file("smalldata/glm_test/binomial_20_cols_10KRows.csv"));
      train.replace((20), train.vec(20).toCategoricalVec()).remove();
      DKV.put(train);
      String[] ignoredCols = new String[]{"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
      String[][] gamCols = new String[][]{{"C11", "C12", "C13"}, {"C14", "C15", "C16"}, {"C17", "C18", "C19"}};
      GAMParameters params = new GAMParameters();
      int k = 12;
      params._response_column = "C21";
      params._ignored_columns = ignoredCols;
      params._num_knots = new int[]{k, k, k};
      params._gam_columns = gamCols;
      params._bs = new int[]{1, 1, 1};
      params._scale = new double[]{10, 10, 10};
      params._train = train._key;
      params._savePenaltyMat = true;
      params._lambda_search = true;
      GAMModel gam = new GAM(params).trainModel().get();
      Scope.track_generic(gam);
      // check starT is of size k x M
      assertTrue((gam._output._starT[0].length == k) && (gam._output._starT[0][0].length == params._M[0]));
      // check penalty_CS is size k x k
      assertTrue((gam._output._penaltyMatCS[0].length == (k - params._M[0])) &&
              (gam._output._penaltyMatCS[0][0].length == (k - params._M[0])));
      // check and make sure polynomials are generated correctly by checking starT
      for (int gamInd = 0; gamInd < gamCols.length; gamInd++) {
        assertCorrectStarT(gam._output._starT[gamInd], gam._output._knots[gamInd], gamCols[gamInd]);
      }
    } finally {
      Scope.exit();
    }
  }
  
  // check correct starT generation for one tp smoother
  public static void assertCorrectStarT(double[][] starT, double[][] knots, String[] gam_columns) {
    int d = gam_columns.length;
    int m = calculatem(d);
    int M = calculateM(d, m);
    int numKnots = knots[0].length;
    int[][] allPolyBasis = convertList2Array(findPolyBasis(d, m), M, d);
    // manually generate starT here and then compare with starT from argument
    double[][] starTManual = new double[numKnots][M];
    for (int rowIndex = 0; rowIndex < numKnots; rowIndex++) {
      for (int colIndex = 0; colIndex < M; colIndex++) {
        starTManual[rowIndex][colIndex] = generate1PolyRow(knots, allPolyBasis[colIndex], rowIndex);
      }
    }
    checkDoubleArrays(starT, starTManual, MAGEPS);
  }
  
  public static double generate1PolyRow(double[][] knots, int[] onePolyBasis, int rowIndex) {
    double temp = 1;
    int d = onePolyBasis.length;
    for (int predInd = 0; predInd < d; predInd++) {
      temp *= Math.pow(knots[predInd][rowIndex], onePolyBasis[predInd]);
    }
    return temp;
  }

  // test with various data transform (standardize, demean) with CS and TP smoothers
  @Test
  public void testDataTransform() {
    Scope.enter();
    try {
      Frame train = Scope.track(parse_test_file("smalldata/glm_test/gaussian_20cols_10000Rows.csv"));
      String[] ignoredCols = new String[]{"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12",
              "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20"};
      String[][] gamCols = new String[][]{{"C11", "C12"}, {"C13", "C14", "C15"}, {"C16"}, {"C17"}};
      GAMParameters params = new GAMParameters();
      params._response_column = "C21";
      params._ignored_columns = ignoredCols;
      params._num_knots = new int[]{5, 11, 4, 10};
      params._gam_columns = gamCols;
      params._bs = new int[]{1, 1, 1, 0};
      params._scale = new double[]{10, 10, 10, 10};
      params._train = train._key;
      params._savePenaltyMat = true;
      params._lambda_search = true;
      //params._standardize = true;
      GAMModel gamStandardize = new GAM(params).trainModel().get();
      Scope.track_generic(gamStandardize);
    } finally {
      Scope.exit();
    }
  }
  
  // test to make sure default knots are generated correctly.
  @Test
  public void testKnotsDefault() {
    Scope.enter();
    try {
      Frame train = Scope.track(parse_test_file("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv"));
      train.replace((10), train.vec(10).toCategoricalVec()).remove();
      DKV.put(train);
      String[] ignoredCols = new String[]{"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
      String[][] gamCols = new String[][]{{"C6"},{"C7", "C8"}, {"C9"}};
      GAMParameters params = new GAMParameters();
      int k = 10;
      params._response_column = "C11";
      params._ignored_columns = ignoredCols;
      params._num_knots = new int[]{0,k,0};
      params._gam_columns = gamCols;
      params._train = train._key;
      params._savePenaltyMat = true;
      params._lambda = new double[]{10};
      GAMModel gam = new GAM(params).trainModel().get();
      // check starT is of size k x M
      assertTrue((gam._output._starT[0].length == k) && (gam._output._starT[0][0].length == params._M[0]));
      // check penalty_CS is size k x k
      assertTrue((gam._output._penaltyMatCS[0].length == (k-params._M[0])) &&
              (gam._output._penaltyMatCS[0][0].length == (k-params._M[0])));
      Scope.track_generic(gam);
    } finally {
      Scope.exit();
    }
  }
  
  // test correct knot generation from Frames for both CS and TP smoothers
  @Test
  public void testKnotsGenerationFromFrame() {
    Scope.enter();
    try {
      Frame knotsFrame1 = generate_real_only(1, 5, 0);
      final double[][] knots = new double[][]{{-1.9990569949269443}, {-0.9814307533427584}, {0.025991586992542004},
              {1.0077098743127828}, {1.999422899675758}};
      new ArrayUtils.CopyArrayToFrame(0,0,knots.length, knots).doAll(knotsFrame1);
      DKV.put(knotsFrame1);
      Scope.track(knotsFrame1);
      int k = 5;
      Frame knotsFrame2 = generate_real_only(2, 5, 0);
      final double[][] knots2 = new double[][]{{0.902652813684858, 1.238501303835733}, {-0.8377150962015311, 
              0.7809874931015846}, {1.0513133931023009, 0.7790618752739205}, {1.9201968414753283, -1.5318363005905211},
              {0.5654843500702142, -1.6560180317092057}};
      new ArrayUtils.CopyArrayToFrame(0,1,knots2.length, knots2).doAll(knotsFrame2);
      DKV.put(knotsFrame2);
      Scope.track(knotsFrame2);

      Frame knotsFrame3 = generate_real_only(1, 7, 0);
      final double[][] knots3 = new double[][]{{-1.9990569949269443}, {-0.9814307533427584}, {0.025991586992542004},
              {0.03}, {0.06}, {1.0077098743127828}, {1.999422899675758}};
      new ArrayUtils.CopyArrayToFrame(0,0,knots3.length, knots3).doAll(knotsFrame3);
      DKV.put(knotsFrame3);
      Scope.track(knotsFrame3);

      Frame train = Scope.track(parse_test_file("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv"));
      String[] ignoredCols = new String[]{"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
      String[][] gamCols = new String[][]{{"C6"},{"C7", "C8"}, {"C9"}};
      train.replace((10), train.vec(10).toCategoricalVec()).remove();
      DKV.put(train);
      GAMParameters params = new GAMParameters();
      params._knot_ids = new String[]{knotsFrame1._key.toString(), knotsFrame2._key.toString(), knotsFrame3._key.toString()};
      params._bs = new int[]{0,1,0};
      params._response_column = "C11";
      params._ignored_columns = ignoredCols;
      params._gam_columns = gamCols;
      params._train = train._key;
      params._savePenaltyMat = true;
      GAMModel gam = new GAM(params).trainModel().get();
      // check starT is of size k x M
      assertTrue((gam._output._starT[0].length == k) && (gam._output._starT[0][0].length == params._M[0]));
      // check penalty_CS is size k x k
      assertTrue((gam._output._penaltyMatCS[0].length == (k-params._M[0])) &&
              (gam._output._penaltyMatCS[0][0].length == (k-params._M[0])));
      Scope.track_generic(gam);
    } finally {
      Scope.exit();
    }
  }

  // test correct distance measurement of thin plate regression smoothers by checking penalty matrix before any kind of
  // transformation.
  @Test
  public void testDistanceMeasure() {
    Scope.enter();
    try {
      Frame train = Scope.track(parse_test_file("smalldata/glm_test/binomial_20_cols_10KRows.csv"));
      train.replace((20), train.vec(20).toCategoricalVec()).remove();
      DKV.put(train);
      String[] ignoredCols = new String[]{"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
      String[][] gamCols = new String[][]{{"C11"},{"C12", "C13"}, {"C14", "C15", "C16"}};
      GAMParameters params = new GAMParameters();
      params._bs = new int[]{1,1,1};
      params._response_column = "C21";
      params._ignored_columns = ignoredCols;
      params._gam_columns = gamCols;
      params._train = train._key;
      params._savePenaltyMat = true;
      GAMModel gam = new GAM(params).trainModel().get();
      Scope.track_generic(gam);
      for (int gamInd = 0; gamInd < gamCols.length; gamInd++)
        assertCorrectDistance(gam._output._penaltyMatrices[gamInd], gam._output._knots[gamInd], gamCols[gamInd]);
    } finally {
      Scope.exit();
    }
  }
  
  // check the calculation of distance for one smoother at a time
  public static void assertCorrectDistance(double[][] penaltyMat, double[][] knots, String[] gamCols) {
    int d = gamCols.length;
    int m = calculatem(d);
    int numKnots = knots[0].length;
    double[][] penaltyMatManual = new double[numKnots][numKnots];
    for (int rowIndex = 0; rowIndex < numKnots; rowIndex++) {
      for (int colIndex = 0; colIndex < numKnots; colIndex++) {
        penaltyMatManual[rowIndex][colIndex] = calDistance(knots, rowIndex, colIndex, d, m);
      }
    }
    checkDoubleArrays(penaltyMat, penaltyMatManual, MAGEPS);
  }
  
  public static double calDistance(double[][] knots, int rowIndex, int colIndex, int predNum, int m) {
    double temp = 0, constant = 1;
    for (int predInd = 0; predInd < predNum; predInd++) { // calculate distance
      double diff = knots[predInd][rowIndex]-knots[predInd][colIndex];
      temp += diff*diff;
    }
    double distance = Math.sqrt(temp);
    // calculate constant to multiply distance
    if (predNum % 2 == 0) { // d is even
      constant = Math.pow(-1, m+1+predNum/2)/(Math.pow(2, m-1)*Math.pow(Math.PI, predNum/2)*factorial(m-predNum));
      if (distance != 0)
        temp = constant * distance * Math.log(distance);
      else 
        temp = 0;
    } else {  // d is odd
      constant = Math.pow(-4, m)*factorial(m)*Math.sqrt(Math.PI)/(factorial(2*m)*Math.pow(2,2*m)*Math.pow(Math.PI, 
              predNum/2.0)*factorial(m-1));
      temp = constant * distance;
    }
    return temp;
  }
  
  // check correct multiplication of zCS and z on data frame
  @Test
  public void testTransformData() {
    Scope.enter();
    try {
      Frame train = Scope.track(parse_test_file("smalldata/glm_test/gaussian_20cols_10000Rows.csv"));
      String[] ignoredCols = new String[]{"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12",
              "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20"};
      String[][] gamCols = new String[][]{{"C13", "C14", "C15"}, {"C20"}, {"C11", "C12"}};
      GAMParameters params = new GAMParameters();
      int k = 10;
      params._response_column = "C21";
      params._ignored_columns = ignoredCols;
      params._num_knots = new int[]{11, k, k};
      params._gam_columns = gamCols;
      params._bs = new int[]{1, 1, 1};
      params._scale = new double[]{1, 1, 1};
      params._train = train._key;
      params._savePenaltyMat = true;
      params._keep_gam_cols = true;
      params._lambda_search = true;
      GAMModel gam = new GAM(params).trainModel().get();
      Scope.track_generic(gam);
      Frame dataFrame = DKV.getGet(gam._output._gamTransformedTrainCenter); // transformed GAM columns from gam model
      Scope.track(dataFrame);
      for (int gamInd = 0; gamInd < gamCols.length; gamInd++)
        assertCorrectTransform(train, dataFrame, 0.2, gamInd, params, gam._output);
    } finally {
      Scope.exit();
    }
  }
  
  // check correct transformation of data frame by zCS and z for one smoother at a time
  public static void assertCorrectTransform(Frame data, Frame gamCols, double frac2Test, int gamIndex, 
                                            GAMParameters parms, GAMModelOutput output) {
    int numKnot = parms._num_knots_sorted[gamIndex]; // number of gam columns to check
    int d = parms._gamPredSize[gamIndex];
    int m = calculatem(d);
    int M = calculateM(d, m);
    int finalFrameCol = numKnot - 1;
    int numKnotsMinusM = numKnot - M;
    int rowNum2Check = (int) Math.floor(data.numRows()*frac2Test);
    int rowInd = Math.round(data.numRows()/rowNum2Check);
    double[][] knots = output._knots[gamIndex];
    ThinPlateDistanceWithKnots tpDistance = new ThinPlateDistanceWithKnots(knots, d, output._oneOGamColStd[gamIndex]);
    int[][] allPolyBasis = convertList2Array(findPolyBasis(d, m), M, d);
    double[] dataInput = new double[d]; // store predictor data before gamification
    double[] dataDistance = new double[numKnot];  // store data after generating distance
    double[] dataPoly = new double[M];  // store data after applying polynomial basis
    double[] dataDistPlusPoly = new double[numKnot];  // store data combining distance and polynomial basis
    double[][] z = transpose(output._zTranspose[gamIndex]);
    double[][] zCS = transpose(output._zTransposeCS[gamIndex]);
    double[] dataOutput = new double[finalFrameCol];  // store gamificed columns from gam model
    
    for (int rowIndex = 0; rowIndex < data.numRows(); rowIndex = rowIndex+rowInd) {
      grabOneRow(data, dataInput, parms._gam_columns_sorted[gamIndex], rowIndex);
      calculateDistance(dataDistance, dataInput, numKnot, knots, d, (d % 2==0), tpDistance._constantTerms, 
              output._oneOGamColStd[rowIndex]);
      double[] dataDistanceCS = multVecArr(dataDistance, zCS);
      generatePolyOneRow(dataInput, allPolyBasis, dataPoly);
      System.arraycopy(dataDistanceCS, 0, dataDistPlusPoly, 0, numKnotsMinusM);
      System.arraycopy(dataPoly, 0, dataDistPlusPoly, numKnotsMinusM, M);
      double[] dataCenterManual = multVecArr(dataDistPlusPoly, z);
      grabOneRow(gamCols, dataOutput, output._gamColNames[gamIndex], rowIndex); // grab gamified columns from gam
      // check to make sure the manually generated gamified rows and the gamified columns from gam model
      checkArrays(dataOutput, dataCenterManual, MAGEPS);
    }
  }
  
  public static void grabOneRow(Frame data, double[] storeOneRow, String[] gamCols, int rowIndex) {
    int d = gamCols.length;
    for (int colIndex = 0; colIndex < d; colIndex++) {
      storeOneRow[colIndex] = data.vec(gamCols[colIndex]).at(rowIndex);
    }
  }

  public static void generatePolyOneRow(double[] dataInput, int[][] onePolyBasis, double[] dataPolyOut) {
    int d = dataInput.length;
    int M = onePolyBasis.length;
    for (int polyInd = 0; polyInd < M; polyInd++) {
      dataPolyOut[polyInd] = 1.0;
      for (int predInd = 0; predInd < d; predInd++) {
        dataPolyOut[polyInd] *= Math.pow(dataInput[predInd], onePolyBasis[polyInd][predInd]);
      }
    }
  }
  
  // check correct multiplication of zCS, z on penalty matrix
  @Test
  public void testTransformPenaltyMatrix() {
    Scope.enter();
    try {
      Frame train = Scope.track(parse_test_file("smalldata/glm_test/binomial_20_cols_10KRows.csv"));
      train.replace((20), train.vec(20).toCategoricalVec()).remove();
      DKV.put(train);
      String[] ignoredCols = new String[]{"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
      String[][] gamCols = new String[][]{{"C11"},{"C12", "C13"}, {"C14", "C15", "C16"}};
      GAMParameters params = new GAMParameters();
      params._bs = new int[]{1,1,1};
      params._response_column = "C21";
      params._ignored_columns = ignoredCols;
      params._gam_columns = gamCols;
      params._train = train._key;
      params._savePenaltyMat = true;
      GAMModel gam = new GAM(params).trainModel().get();
      Scope.track_generic(gam);
      for (int gamInd = 0; gamInd < gamCols.length; gamInd++)
        assertCorrectTransform(gam._output._zTransposeCS[gamInd], gam._output._zTranspose[gamInd], 
                gam._output._penaltyMatrices[gamInd], gam._output._penaltyMatCS[gamInd], 
                gam._output._penaltyMatricesCenter[gamInd]);
    } finally {
      Scope.exit();
    }
  }
  
  public static void assertCorrectTransform(double[][] zCST, double[][] zT, double[][] penaltyMat, 
                                            double[][] penaltyMatCS, double[][] penaltyMatCenter) {
    int numRow = penaltyMat.length;
    int numCol = penaltyMat[0].length;
    double[][] zCS = ArrayUtils.transpose(zCST);
    // check the application of zCS to penalty matrix is correct;
    double[][] penaltyMatCSManual = multArrArr(multArrArr(zCST, penaltyMat), zCS);
    checkDoubleArrays(penaltyMatCS, penaltyMatCSManual, MAGEPS);    
    double[][] penaltyMatCSManualExpand = expandArray(penaltyMatCSManual, zT[0].length);
    double[][] z = transpose(zT);
    double[][] penaltyMatCenterManual = multArrArr(multArrArr(zT, penaltyMatCSManualExpand), z);
    // check the application of zCS to penalty matrix CS is correct;
    checkDoubleArrays(penaltyMatCenter, penaltyMatCenterManual, MAGEPS);
  }
  
  // For a given d, calculate m, then calculate the polynomial basis degree for each predictor involves.
  // However, the 0th order is not included at this stage.
  @Test
  public void testFindPolybasis() {
    int[] d = new int[]{1, 3, 5, 8, 10};
    int[] ans = new int[]{2, 10, 56, 495, 3003};
    for (int index = 0; index < d.length; index++) {
      int m = calculatem(d[index]);
      int[] polyOrder = new int[m];
      for (int tempIndex = 0; tempIndex < m; tempIndex++)
        polyOrder[tempIndex] = tempIndex;
      List<Integer[]> polyBasis = findPolyBasis(d[index], m);
      assertEquals(ans[index], polyBasis.size()); // check and make sure number of basis is correct.  Content checked in testFindPermManyD already
      assertCorrectAllPerms(polyBasis, polyOrder);
    }
  }
  
  // given one combination, test that all permutations are returned
  @Test
  public void testFindAllPolybasis() {
    List<Integer[]> listOfCombos = new ArrayList<>();
    listOfCombos.add(new Integer[]{0, 0, 0, 0, 1}); // should get 5 permutations
    listOfCombos.add(new Integer[]{1, 2, 0, 0, 0}); // should get 20 permutations
    List<Integer[]> allCombos = findAllPolybasis(listOfCombos); // should be of size 5+20+1 (from all zeroes)
    assertEquals(26, allCombos.size()); // check correct size
    assertCorrectAllPerms(allCombos, new int[]{0,1,3}); // check correct content
  }
  
  public static void assertCorrectAllPerms(List<Integer[]> allCombos, int[] correctVals) {
    for (Integer[] oneList : allCombos) {
      int sumVal = sum(Arrays.stream(oneList).mapToInt(Integer::intValue).toArray());
      boolean correctSum = false;
      for (int val : correctVals)
        correctSum = correctSum || (sumVal == val);
      assertTrue(correctSum);
    }
  }
  
  @Test
  public void testFindPermManyD() {
    int[] d = new int[]{1, 3 ,5, 8, 10};
    int[] correctComboNum = new int[]{1, 3, 6, 11, 18};
    for (int index = 0; index < d.length; index++) {
      testFindPerm(d, correctComboNum, index);
    }
  }
  
  public void testFindPerm(int[] d, int[] correctComboNum, int testIndex) {
    int m = calculatem(d[testIndex]); // highest order of polynomial basis is m-1
    int[] totDegrees = new int[m];
    int[] degreeCombos = new int[m-1];
    for (int index = 0; index < totDegrees.length; index++)
      totDegrees[index] = index;
    int count = 0;
    for (int index = m-1; index > 0; index--) {
      degreeCombos[count++] = index;
    }

    // check for combos for totDegree = 0, 1, ..., m-1
    int numCombo = 0;
    for (int degree : totDegrees) {
      ArrayList<int[]> allCombos = new ArrayList<>();
      findOnePerm(degree, degreeCombos, 0, allCombos, null);
      assertCorrectPerm(allCombos, degree, degreeCombos);
      numCombo += allCombos.size();
    }
    assertEquals(numCombo, correctComboNum[testIndex]); // number of combos are correct
  }
  
  public static void assertCorrectPerm(ArrayList<int[]> allCombos, int degree, int[] degreeCombos) {
    for (int index = 0; index < allCombos.size(); index++) {
      int[] oneCombo = allCombos.get(index);
      int sum = 0;
      for (int tmpIndex = 0; tmpIndex < degreeCombos.length; tmpIndex++) {
        sum += oneCombo[tmpIndex]*degreeCombos[tmpIndex];
      }
      assertEquals(degree, sum);
    }
  }
  
  @Test
  public void testCalculatem() {
    int[] d = new int[]{1, 2, 3, 4, 10};
    int[] ans = new int[]{2, 2, 3, 3, 6};  // calculated by using (floor(d+1)/2)+1 from R
    
    for (int index = 0; index < d.length; index++) {
      int m = calculatem(d[index]);
      assertEquals(m, ans[index]);
    }
  }

  @Test
  public void testCalculateM() {
    int[] d = new int[]{1, 2, 3, 4, 5};
    int[] m = new int[]{2, 2, 3, 3, 4};  // calculated by using (floor(d+1)/2)+1 from R

    for (int index = 0; index < d.length; index++) {
      int M = calculateM(d[index], m[index]);
      assertEquals(M, factorial(d[index]+m[index]-1)/(factorial(d[index])*factorial(m[index]-1)));
    }
  }
  
  public static int factorial(int n) {
    int prod = 1; 
    for (int index = 1; index <= n; index++)
      prod *= index;
    return prod;
  }
}
