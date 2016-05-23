// std
#include <iostream>
#include <vector>
#include <string>

// Eigen
#include <eigen3/Eigen/Dense>

// Boost
#include <boost/filesystem.hpp>

// OpenCV
#include <opencv2/opencv.hpp>

// VLFeat 
// The VLFeat header files need to be declared external.
extern "C" {
  #include <vl/generic.h>
  #include <vl/dsift.h>
  #include <vl/gmm.h>
  #include <vl/fisher.h>
  #include <vl/svm.h>
}

struct gmmModel
{
  Eigen::MatrixXd means;
  Eigen::MatrixXd covs;
  Eigen::MatrixXd priors;
  int Nclusters;
  int dimension;
};


// Some helper functions for writing and reading files.
void convVec2EigenMat(const std::vector<Eigen::MatrixXd> &eigMatVec, Eigen::MatrixXd &X );
int writeMatrix2File(const Eigen::MatrixXd &matrix, const std::string &fileName);
int readMatrixFromFile(const std::string &fileName, Eigen::MatrixXd &X, const int &reservSize=1E6);
void getListOfFilesInDir(const boost::filesystem::path& root, const std::vector<std::string>& ext, std::vector<boost::filesystem::path> &ret, const bool fullPath);
void writeVecToFile(const std::string &fileName, const std::vector<int> &vec, const std::string &delimeter);
void printMatSize(const Eigen::MatrixXd &X, const std::string &MatName);

// Main functions
void getFileNamesAndLabels(const std::vector<std::string> labelNames, const std::string &dataPath, std::vector<std::string> &fileList, std::vector<int> &labels, const bool &verbose=false);
void computeDSift(const std::vector<std::string> &fileList, std::vector<Eigen::MatrixXd> &siftMatVec, const bool &verbose=false);
void computeGMM(const std::vector<Eigen::MatrixXd> &siftMatVec, gmmModel &gmmParams, const bool &verbose=false);
void computeFisherVector(const std::vector<Eigen::MatrixXd> &siftMatVec, const gmmModel &gmmParams, Eigen::MatrixXd &Xfisher, const bool &verbose=false);
void trainSVM(const Eigen::MatrixXd &Xfisher, const std::vector<int> &labels, const int &Nuniquelabels, Eigen::MatrixXd &svmModels, const bool &verbose=false);
void predict(const Eigen::MatrixXd &svmModels, Eigen::MatrixXd &X, std::vector<int> &predictedLabels, const bool &verbose=false);
void computePCA(const Eigen::MatrixXd &X, Eigen::MatrixXd &projMat, const int &projDim, const bool &scalingOn=false );
void colwiseMinMaxIdx(const Eigen::MatrixXd &X, Eigen::MatrixXd &idxs, const bool findMaxIdx=true);

int main (int argc, const char * argv[])
{

  // Figure out if we are training or predicting...
  std::string todo = "";
  if(argc > 1) todo = std::string(argv[1]);

  bool doPredict = false;
  std::string fName = "";
  if(todo.compare("predict")==0)
  {
    doPredict = true;
    std::cout << "\n\033[92m" << "# Running prediction..." << "\033[0m\n" << std::flush;
  }
  else if(todo.compare("predict")==0)
  {
    std::cout << "\n\033[92m" << "# Training model..." << "\033[0m\n" << std::flush;
  }
  else
  {
    std::cout << "\n\033[92m" << "You need to provide argument: train or predict!" << "\033[0m\n" << std::flush;
    return 0;
  }


  // Set up data paths we should enable this as a argument instead. 
  std::string dataTrainPath = "train";
  std::string dataTestPath = "test";
  std::vector<std::string> labelNames = {"glass","carton","porcelain","metal","plastic","wood"};
  int projDim = 80; // PCA projection dimension


  if(todo.compare("train")==0)
  {       

    // ### GET LIST OF FILES AND THEIR CORRESPONDING LABEL
    // Use image names as label assignments
    std::vector<int> labels;
    std::vector<std::string> fileList;
    getFileNamesAndLabels(labelNames,dataTrainPath,fileList,labels,false);

    
    // ### COMPUTE COMPUTE DENSE SIFT FOR EACH IMAGE
    std::vector<Eigen::MatrixXd> siftMatVec;
    computeDSift(fileList,siftMatVec,true);


    // ### COMPUTE PCA PROJECTION
    std::cout << "\n\033[92m" << "# Computing PCA..." << "\033[0m\n" << std::flush;
    // I know this is ugly...
    Eigen::MatrixXd pcaProjMat;
    Eigen::MatrixXd Xsifts;
    convVec2EigenMat(siftMatVec,Xsifts);    
    computePCA(Xsifts,pcaProjMat,projDim);
    // Free HUGE sift matrix
    Xsifts.resize(0,0);
    writeMatrix2File(pcaProjMat, "pcaProjMat.txt");
    // Eigen::MatrixXd projMat;
    //readMatrixFromFile("pcaProjMat.txt",projMat);
    // Compute PCA projection for each img sift representation
    for(int idx = 0; idx < siftMatVec.size(); idx++)
      siftMatVec[idx] = siftMatVec[idx] * pcaProjMat;



    // ### COMPUTE GMM MODEL
    gmmModel gmmParams;
    gmmParams.Nclusters = 256;
    gmmParams.dimension = 80;
    computeGMM(siftMatVec, gmmParams, true);
    writeMatrix2File(gmmParams.means, "gmmMeans.txt");
    writeMatrix2File(gmmParams.covs, "gmmCovs.txt");
    writeMatrix2File(gmmParams.priors, "gmmPriors.txt");
    

    // Read gmm parameters from file
    // gmmModel gmmParams;
    // gmmParams.Nclusters = 256;
    // gmmParams.dimension = 80;       
    // readMatrixFromFile("gmmMeans.txt",gmmParams.means);
    // readMatrixFromFile("gmmCovs.txt",gmmParams.covs);
    // readMatrixFromFile("gmmPriors.txt",gmmParams.priors);


    
    // ### COMPUTE FISHER VECTORS
    Eigen::MatrixXd Xfisher(fileList.size(),2 * gmmParams.dimension * gmmParams.Nclusters);
    computeFisherVector(siftMatVec, gmmParams, Xfisher);
    writeMatrix2File(Xfisher, "FisherVector.txt");
    // Eigen::MatrixXd Xfisher;
    // readMatrixFromFile("FisherVector.txt",Xfisher);



    // ### GET CAFFE VECTORS 
    Eigen::MatrixXd Xcaffe;
    readMatrixFromFile("caffe_train.txt",Xcaffe);


    // ### CONCATENATE THE FISHER AND THE CAFFE VECTOR
    Eigen::MatrixXd Xfc(fileList.size(),Xfisher.cols()+Xcaffe.cols());
    Xfc << Xfisher, Xcaffe;
    // writeMatrix2File(Xfc, "FisherCafffeVec.txt");


    // ### TRAIN SVM MODELS 
    // Store them in eigen matrix [w b]. We compute scores = w'*xtest + b ;
    Eigen::MatrixXd svmModels(labelNames.size(),Xfc.cols()+1);
    trainSVM(Xfc, labels, labelNames.size(), svmModels);
    writeMatrix2File(svmModels, "svmModels_fishercaffe.txt");
    // Eigen::MatrixXd svmModels;
    // readMatrixFromFile("svmModel.txt",Xcaffe);

    
    // ### DO PREDICTION
    std::vector<int> predictedLabels;
    predict(svmModels, Xfc, predictedLabels);
    
    std::cout << "\n\033[92m" << "# Final Predictions..." << "\033[0m\n" << std::flush;
    int Ncorrect = 0;
    for(int idx = 0; idx < predictedLabels.size(); idx++)
    {
      if(predictedLabels[idx]==labels[idx]) Ncorrect +=1;
      std::cout << " Predicted label: " << labelNames[predictedLabels[idx]] << " for file with label " << labelNames[labels[idx]] << std::endl; //<< " and file name " << fileList[idx] << std::endl;
    }

    std::cout << "Percent Correct: " << 100*( (float)Ncorrect/(float)predictedLabels.size() ) << std::endl;

  }
  else
  {
    // Do prediction with learned model
 
    // ### Get test files
    std::vector<int> labels;
    std::vector<std::string> fileList;
    getFileNamesAndLabels(labelNames,dataTestPath,fileList,labels,false);
    
    // Create matrices for pipeline
    Eigen::MatrixXd pcaProjMat;
    readMatrixFromFile("pcaProjMat.txt",pcaProjMat);
    gmmModel gmmParams;
    gmmParams.Nclusters = 256;
    gmmParams.dimension = 80;       
    readMatrixFromFile("gmmMeans.txt",gmmParams.means);
    readMatrixFromFile("gmmCovs.txt",gmmParams.covs);
    readMatrixFromFile("gmmPriors.txt",gmmParams.priors);
    Eigen::MatrixXd svmModels;
    readMatrixFromFile("svmModels.txt",svmModels);


    // ### GET CAFFE VECTORS 
    Eigen::MatrixXd Xcaffe;
    readMatrixFromFile("caffe_test.txt",Xcaffe);


    std::vector<int> predictedLabels;
    // Do prediction for each file since we have many files..
    for(int iFile = 0; iFile < fileList.size(); iFile++)
    {

      // ### COMPUTE COMPUTE DENSE SIFT FOR IMAGE
      std::vector<std::string> file = {fileList[iFile]};
      std::vector<Eigen::MatrixXd> siftMatVec;
      computeDSift(file,siftMatVec,false);

      // ### PROJECT ONTO PRINCIPAL AXES
      for(int idx = 0; idx < siftMatVec.size(); idx++)
        siftMatVec[idx] = siftMatVec[idx] * pcaProjMat;

      // ### COMPUTE FISHER VECTORS   
      Eigen::MatrixXd Xfisher(1,2 * gmmParams.dimension * gmmParams.Nclusters);
      computeFisherVector(siftMatVec, gmmParams, Xfisher, false); 

      // ### CONCATENATE FISHER AND CAFFE VECTORS
      Eigen::MatrixXd Xfc(1,Xfisher.cols()+Xcaffe.cols());
      Xfc << Xfisher.row(0), Xcaffe.row(iFile);      

      // ### DO PREDICTION 
      std::vector<int> predVec;
      predict(svmModels,Xfc,predVec,false);
      predictedLabels.push_back(predVec[0]);
        
    }

    std::cout << "\n\033[92m" << "# Final Predictions..." << "\033[0m\n" << std::flush;
    int Ncorrect = 0;
    for(int idx = 0; idx < predictedLabels.size(); idx++)
    {
      if(predictedLabels[idx]==labels[idx]) Ncorrect +=1;
      std::cout << " Predicted label: " << labelNames[predictedLabels[idx]] << " for file with label " << labelNames[labels[idx]] << " and file name " << fileList[idx] << std::endl << std::flush;
    }

    std::cout << "Percent Correct: " << 100*( (float)Ncorrect/(float)predictedLabels.size() ) << std::endl;
    
    writeVecToFile("labelpredictions.txt", predictedLabels, "\n");
  }

  return 0;
}


void 
convVec2EigenMat(const std::vector<Eigen::MatrixXd> &eigMatVec, Eigen::MatrixXd &X )
{
  // Copy all matrices in the vec to one big matrix
  // Get size
  int Nrows = 0;
  int Ncols = 0; 
  for(int idx = 0; idx < eigMatVec.size(); idx++)
    Nrows += eigMatVec[idx].rows();

  // Create matrix and copy
  Ncols = eigMatVec[0].cols();
  X = Eigen::MatrixXd::Zero(Nrows,Ncols);

  int rowPos = 0;
  for(int idx = 0; idx < eigMatVec.size(); idx++)
  {
    X.block(rowPos,0,eigMatVec[idx].rows(),Ncols) = eigMatVec[idx];
    rowPos += eigMatVec[idx].rows();    
  }   
}




void 
getFileNamesAndLabels(const std::vector<std::string> labelNames, const std::string &dataPath, std::vector<std::string> &fileList, std::vector<int> &labels, const bool &verbose)
{
  if (verbose) std::cout << "\n\033[92m" << "# Reading all files... " << "\033[0m\n" << std::flush;

  std::vector<boost::filesystem::path> ret;
  boost::filesystem::path p (dataPath);
  std::vector<std::string> exts = {".jpg",".jpeg",".png"};

  getListOfFilesInDir(p, exts, ret,true);

  for(int i_file = 0; i_file < ret.size(); i_file++ )
  {
    // Get label from the specific directory the img is in, that is, all images of one class is 
    // in a specific directory named after the class label. So we do simple string search in the 
    // file path to assign the label. Each class label is assigned a number 0,1,2... and so on.
    labels.push_back(0);
    for(int i = 0; i < labelNames.size(); i++)
    {
      std::size_t found = ret[i_file].string().find(labelNames[i]);
      if (found!=std::string::npos)
      {
        labels[i_file] = i;
        // std::cout << "Pushed label " << i << std::endl;
        break;
      }

    }
    fileList.push_back(ret[i_file].string());
    if (verbose) std::cout << "Adding file: " << ret[i_file].string() << std::endl << std::flush;
  }   

  if (verbose) std::cout << "\033[92m" << "Done. " << "\033[0m\n" << std::flush;
}



void 
computeDSift(const std::vector<std::string> &fileList, std::vector<Eigen::MatrixXd> &siftMatVec, const bool &verbose)
{
  if (verbose)
    std::cout << "\n\033[92m" << "# Computing Dense SIFT for all images..." << "\033[0m\n" << std::flush;

  for(std::vector<std::string>::const_iterator fName = fileList.begin(); fName != fileList.end(); ++fName)
  {   
    if (verbose)
      std::cout << *fName << std::flush << std::endl;
    
    cv::Mat img = cv::imread(*fName,0);
    // transform image in cv::Mat to float vector
    std::vector<float> imgvec;
    imgvec.reserve(img.rows*img.cols);
    for (int i = 0; i < img.rows; ++i)
    {
      for (int j = 0; j < img.cols; ++j)
      {
      imgvec.push_back(img.at<unsigned char>(i,j) / 255.0f);                                                                                                                                                                                                        
      }
    }

    VlDsiftFilter* vlf = vl_dsift_new_basic(img.cols, img.rows, 4, 8);
    vl_dsift_process(vlf, &imgvec[0]);
    img.release();  
    // std::cout << "Number of keypoints: " << vl_dsift_get_descriptor_size(vlf) << std::flush << std::endl;
    // std::cout  << "Descriptor size: " << vl_dsift_get_keypoint_num(vlf) << std::flush << std::endl;

    int Nkeypoints = vl_dsift_get_keypoint_num(vlf); //10980
    int descSize = vl_dsift_get_descriptor_size(vlf); //128

    // Copy descriptors to eigen matrix
    float * descArray = (float*) vl_malloc(sizeof(float) * descSize * Nkeypoints ) ;
    descArray = (float*)vl_dsift_get_descriptors(vlf);

    Eigen::MatrixXd keypointsMat = Eigen::MatrixXd::Zero(Nkeypoints,descSize);

    if (verbose) std::cout << "Copying sift\n" << std::flush;    

    // For each descriptor in dense sift
    for(int idxKeypoint = 0; idxKeypoint < Nkeypoints; idxKeypoint++)
      for(int sidx = 0; sidx < descSize; sidx++)
        keypointsMat(idxKeypoint,sidx) = descArray[idxKeypoint*descSize+sidx];

    siftMatVec.push_back(keypointsMat);


    // Free memory
    //delete descArray;
    vl_dsift_delete(vlf);
    imgvec.clear();
  }

  if (verbose) std::cout << "\033[92m" << "Done. " << "\033[0m\n" << std::flush;    
}



void 
computeGMM(const std::vector<Eigen::MatrixXd> &siftMatVec, gmmModel &gmmParams, const bool &verbose)
{
  if(verbose) std::cout << "\n\033[92m" << "# Computing GMM..." << "\033[0m\n" << std::flush;
  if(verbose) std::cout << "Initing..." << std::endl << std::flush;
  
  Eigen::MatrixXd Xsifts;
  convVec2EigenMat(siftMatVec,Xsifts);    

  // Map feature eigen matrix to C float
  // The transpose is there since Eigen saves in column major and vlfeat assummens features 
  // stacked ontop of each other. If we didn't do transpose we would have X_11 X_21 X_31... 
  // instead of X_11 X_12 X_13
  double * Xdouble = (double*) vl_malloc(sizeof(double) * Xsifts.rows() * Xsifts.cols() ) ;
  Eigen::Map<Eigen::MatrixXd>( Xdouble, Xsifts.cols(), Xsifts.rows() ) = Xsifts.transpose();

  // Compute the GMM
  vl_size dimension = Xsifts.cols();
  int Npts = Xsifts.rows();  
  int numClusters = gmmParams.Nclusters;  

  VlGMM* gmm = vl_gmm_new (VL_TYPE_DOUBLE, dimension, numClusters) ;
  vl_gmm_set_max_num_iterations (gmm, 100) ;
  vl_gmm_set_initialization (gmm,VlGMMKMeans);
  int level = 0;
  if(verbose) level = 2;
  vl_gmm_set_verbosity (gmm,level);

  if(verbose) std::cout << "Running..." << std::endl << std::flush;
  vl_gmm_cluster (gmm, Xdouble, Xsifts.rows() );

  // Save the params to struct mean and covariance
  gmmParams.means = Eigen::Map<Eigen::MatrixXd>( (double*) vl_gmm_get_means(gmm), 1, dimension*numClusters);
  gmmParams.covs = Eigen::Map<Eigen::MatrixXd>( (double*) vl_gmm_get_covariances(gmm), 1, dimension*numClusters);
  gmmParams.priors = Eigen::Map<Eigen::MatrixXd>( (double*) vl_gmm_get_priors(gmm), 1, numClusters);

  if(verbose) std::cout << "Done..." << std::endl << std::flush;

}


void 
computeFisherVector(const std::vector<Eigen::MatrixXd> &siftMatVec, const gmmModel &gmmParams, Eigen::MatrixXd &Xfisher, const bool &verbose)
{

  if(verbose) std::cout << "\n\033[92m" << "# Encoding Fisher vectors..." << "\033[0m\n" << std::flush;

  vl_size dimension = gmmParams.dimension;
  vl_size numClusters = gmmParams.Nclusters;

  // Map gmm parameters eigen vectors to vlfeat variables
  double * gmmMeans = (double*) vl_malloc(sizeof(double) * gmmParams.means.size()) ;
  Eigen::Map<Eigen::MatrixXd>( gmmMeans, 1, gmmParams.means.size() ) = gmmParams.means;  

  double * gmmCovs = (double*) vl_malloc(sizeof(double) * gmmParams.covs.size()) ;
  Eigen::Map<Eigen::MatrixXd>( gmmCovs, 1, gmmParams.covs.size() ) = gmmParams.covs; 

  double * gmmPriors = (double*) vl_malloc(sizeof(double) * gmmParams.priors.size()) ;
  Eigen::Map<Eigen::MatrixXd>( gmmPriors, 1, gmmParams.priors.size() ) = gmmParams.priors;


  // For all instances
  for(int idx = 0; idx < siftMatVec.size(); idx++)
  {       
    double * X = (double*) vl_malloc(sizeof(double) * siftMatVec[idx].rows() * siftMatVec[idx].cols() ) ;
    Eigen::Map<Eigen::MatrixXd>( X, siftMatVec[idx].cols(), siftMatVec[idx].rows() ) = siftMatVec[idx].transpose();

    double* enc = (double*) vl_malloc(sizeof(double) * 2 * dimension * numClusters);
    vl_fisher_encode(enc, VL_TYPE_DOUBLE, gmmMeans, dimension,
             numClusters, gmmCovs, gmmPriors,
             X, siftMatVec[idx].rows(), VL_FISHER_FLAG_IMPROVED);

    for(int iVal = 0; iVal < 2 * dimension * numClusters; iVal++)
      Xfisher(idx,iVal) = enc[iVal];
    
    vl_free(enc);
  }

  if (verbose)
    std::cout << "\033[92m" << "Done. " << "\033[0m\n" << std::flush;  
}



void 
trainSVM(const Eigen::MatrixXd &Xfisher, const std::vector<int> &labels, const int &Nuniquelabels, Eigen::MatrixXd &svmModels, const bool &verbose)
{
  if(verbose) std::cout << "\n\033[92m" << "# Trainging SVMs - one against many..." << "\033[0m" << std::flush;   

  vl_size const numData = Xfisher.rows(); ;
  vl_size const dimensions = Xfisher.cols();

  // Map Eigen vector to array
  double * xTr = (double*) vl_malloc(sizeof(double) * Xfisher.rows() * Xfisher.cols() ) ;
  Eigen::Map<Eigen::MatrixXd>( xTr, Xfisher.cols(), Xfisher.rows() ) = Xfisher.transpose();

  // Train the SVM 1 against k
  for(int label = 0; label < Nuniquelabels; label++)
  {
    // Assign  1 to current class and -1 to others
    double y[labels.size()];
    for(int lidx = 0; lidx < labels.size(); lidx++)
    {
      if(labels[lidx]==label)
        y[lidx] = 1;
      else
        y[lidx] = -1;
    }

    
    double Ntrain = (double)labels.size();
    double C = 10.;
    double lambda = 1. / (C*Ntrain);
    double * model ;
    double bias ;
    VlSvm * svm = vl_svm_new(VlSvmSolverSgd,xTr,dimensions,numData,y,lambda);
    vl_svm_train(svm) ;

    model = (double*) vl_svm_get_model(svm);
    bias = vl_svm_get_bias(svm);

    // Copy linear svm model to Eigen matrix
    for(int i = 0; i < Xfisher.cols(); i++)
      svmModels(label,i) = model[i];
    
    svmModels(label, Xfisher.cols()) = bias;

    vl_svm_delete(svm) ;
  }

  if(verbose) std::cout << "\n\033[92m" << "Done." << "\033[0m\n" << std::flush;  
}



void 
predict(const Eigen::MatrixXd &svmModels, Eigen::MatrixXd &X, std::vector<int> &predictedLabels, const bool &verbose)
{
  if(verbose)
  {
    std::cout << "\n\033[92m" << "# Classifying test data" << "\033[0m\n" << std::flush;
    printMatSize(svmModels,"svmModels");
    printMatSize(X,"X");
  }

  // Compute s = w*x.T for all data -> S = W * X.T. Where W is the weight matrix for each models and X is the data matrix. 
  // S(k x N) = W(k x M) * X(N x M).T
  Eigen::MatrixXd scores(svmModels.rows(),X.rows());
  scores = svmModels.block(0,0,svmModels.rows(),svmModels.cols()-1) * X.transpose();

  // Add bias, s += b
  for(int label = 0; label < svmModels.rows(); label++)
    scores.row(label).array() += svmModels(label,svmModels.cols()-1);
  
  if(verbose)
  { 
    std::cout << "Printing SVM scores...\n" << std::flush;
    std::cout << scores << std::endl;
  }

  // For each column in S(k x N) find the max value and corresponding row index which indicates the label.
  Eigen::MatrixXd idxs = Eigen::MatrixXd::Zero(1,X.rows());
  colwiseMinMaxIdx(scores, idxs);
  if(verbose) std::cout << "Predicted :" << idxs << std::endl;


  for(int i=0; i < idxs.cols(); i++)
    predictedLabels.push_back(idxs(0,i));

  if (verbose)
    std::cout << "\n\033[92m" << "Done. " << "\033[0m\n" << std::flush;  
}


int
writeMatrix2File(const Eigen::MatrixXd &matrix, const std::string &fileName)
{
  std::ofstream out( fileName.c_str() );

  if (out.is_open())
  out << matrix;
  else
  return 0;

  out.close();
  return 1;
}


int 
readMatrixFromFile(const std::string &fileName, Eigen::MatrixXd &X, const int &reservSize)
{
  int cols = 0, rows = 0;
  std::vector<double> buff;
  buff.reserve(reservSize);

  // Read numbers from file into buffer.
  std::ifstream infile;
  infile.open(fileName.c_str());
  if(!infile.is_open())
    return 0;

  while (! infile.eof())
  {
    // For each row in file
    std::string line;
    std::getline(infile, line);

    int temp_cols = 0;
    std::stringstream stream(line);
    while(! stream.eof())
    {
      double number;
      stream >> number;
      temp_cols++;
      buff.push_back( number);
    }
    if (temp_cols == 0)
      continue;

    if (cols == 0)
      cols = temp_cols;

    rows++;
  }

  infile.close();

  // Populate matrix with numbers.
  X = Eigen::MatrixXd::Zero(rows,cols);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      X(i,j) = buff[ cols*i+j ];

  return 1;
}



/* Returns the filenames of all files that have the specified extension
 * in the specified directory and all subdirectories
 */
void
getListOfFilesInDir(const boost::filesystem::path& root, const std::vector<std::string>& ext, std::vector<boost::filesystem::path> &ret, const bool fullPath)
{
  namespace fs = ::boost::filesystem;
  if (!fs::exists(root))
  {
  printf("Directory not found!");
  return;
  }

  if (fs::is_directory(root))
  {
  fs::recursive_directory_iterator it(root);
  fs::recursive_directory_iterator endit;

  while(it != endit)
  {
    if (fs::is_regular_file(*it))
    {
    bool hasExt = false;
    for(int idx = 0; idx < ext.size(); idx++)
      if(it->path().extension() == ext[idx])
      {
      hasExt = true;
      break;
      }
    
    if(hasExt)
    {
      if(fullPath)
      ret.push_back(it->path().string());
      else
      ret.push_back(it->path().filename());
    }
    }
    ++it;
  }
  }
}




void 
computePCA(const Eigen::MatrixXd &X, Eigen::MatrixXd &projMat, const int &projDim, const bool &scalingOn )
{
  
  assert(X.rows() > projDim);

  Eigen::MatrixXd Xprj = X;

  // Make matrix zero mean
  for(int iCol=0; iCol!=Xprj.cols(); iCol++)
  {
    Xprj.col(iCol).array() -= (Xprj.col(iCol)).mean();
  }

  // If scaling is on scale each dimension to variance one.
  if(scalingOn)
  {
    double factor = Xprj.rows()-1;
    for(int iCol=0; iCol!=Xprj.cols(); iCol++)
    {
      Xprj.col(iCol) /= std::sqrt( Xprj.col(iCol).dot(Xprj.col(iCol)) / factor );
    }
  }

  // Divide matrix by sqrt dim -1 so SVD trick will work.
  Xprj /= std::sqrt(Xprj.cols()-1);

  // Do Eigen SVD.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(Xprj, Eigen::ComputeThinU | Eigen::ComputeThinV);

  // Compute projection matrix
  projMat = (svd.matrixV()).block(0,0,(svd.matrixV()).rows(),projDim);

}


void 
colwiseMinMaxIdx(const Eigen::MatrixXd &X, Eigen::MatrixXd &idxs,const bool findMaxIdx)
{
    Eigen::MatrixXf::Index colIdx;
    for(int iCol = 0; iCol < X.cols(); iCol++)
    {
      findMaxIdx ? X.col(iCol).maxCoeff(&colIdx) : X.col(iCol).minCoeff(&colIdx);
      idxs(0,iCol) = colIdx;
    }
}



void
writeVecToFile(const std::string &fileName, const std::vector<int> &vec, const std::string &delimeter)
{
  std::ofstream file(fileName.c_str(), std::ios::out| std::ios::app);
  if (file.is_open())
  {
    for(std::vector<int>::const_iterator iter = vec.begin(); iter != vec.end(); ++iter)
      if(iter+1 != vec.end())
        file << (*iter) << delimeter;
      else 
        file << (*iter);
}
file << std::endl;
file.close();
}


void
printMatSize(const Eigen::MatrixXd &X, const std::string &MatName)
{
  std::cout << MatName << " " << X.rows() << " x " << X.cols() << std::endl;
}
