using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using System;
using System.Drawing;
using System.Reflection.Emit;
using System.Windows.Forms;

namespace sign_lao_langude
{
    public partial class Form1 : Form
    {
        private VideoCapture capture;
        private readonly CascadeClassifier faceCascade;
        private readonly string[] labels = { "ກ", "ຂ", "ຄ", "ງ", "ຈ", "ຊ", "ຍ", "ດ", "ຕ", "ທ", "ນ", "ບ", "ຜ", "ຝ", "ພ", "ຟ", "ມ", "ຣ", "ລ", "ວ", "ສ", "ຫ", "ອ" };
        private const int roiWidth = 250;
        private const int roiHeight = 250;
        private Net net;

        public Form1()
        {
            InitializeComponent();
            capture = new VideoCapture();
            faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");
            Application.Idle += ProcessFrame;

            // Load the ONNX model
            net = DnnInvoke.ReadNetFromONNX("C:\\Users\\Sitth\\source\\repos\\Computer vision\\Homework\\sign_lao_langude\\sign_lao_langude\\model");
        }

        private void button1_Click(object sender, EventArgs e)
        {

        }

        private void Form1_Load(object sender, EventArgs e)
        {
            capture.Start();
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            using (var frame = capture.QueryFrame())
            {
                if (frame != null)
                {
                    // Resize the frame to match the input size expected by the model
                    Mat resizedFrame = new Mat();
                    CvInvoke.Resize(frame, resizedFrame, new Size(224, 224));

                    // Convert the resized frame to a blob
                    Mat blob = DnnInvoke.BlobFromImage(resizedFrame, 1, new Size(224, 224), new MCvScalar(104, 117, 123));

                    // Set the blob as input to the model
                    net.SetInput(blob);

                    // Perform inference
                    Mat prob = net.Forward();

                    // Convert the result to a readable format
                    Mat probMat = prob.Reshape(1, 1); // Reshape the blob to 1x1 matrix
                    float[] resultData = new float[probMat.Cols * probMat.Rows * probMat.NumberOfChannels];
                    probMat.CopyTo(resultData); // Copy the data to an array

                    // Do something with the result data
                    // For example, display it in a label
                    label1.Text = string.Join(", ", resultData);

                    // Convert the frame to bitmap
                    Bitmap bitmap = frame.ToBitmap();

                    // Display the bitmap in the PictureBox
                    pictureBox1.Image = bitmap;

                    // Perform any other processing here...
                }
            }
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            using (var frame = capture.QueryFrame())
            {
                if (frame != null)
                {
                    frame.Flip(Emgu.CV.CvEnum.FlipType.Horizontal);

                    var grayFrame = frame.Convert<Gray, byte>();
                    var faces = faceCascade.DetectMultiScale(grayFrame, 1.1, 3, new System.Drawing.Size(20, 20));

                    foreach (var face in faces)
                    {
                        var roi = frame.GetSubRect(face);
                        var roiBitmap = roi.Bitmap;
                        var avgVariance = CalculateAverageVariance(roiBitmap);

                        if (avgVariance > 1000)
                        {
                            var resizedRoi = roiBitmap.Resize(100, 100, Emgu.CV.CvEnum.Inter.Linear);
                            var prediction = Predict(resizedRoi);
                            var confidence = prediction.Max();
                            var predictedLabelIndex = Array.IndexOf(prediction, confidence);
                            var predictedLabel = labels[predictedLabelIndex];

                            Console.WriteLine($"Prediction: {predictedLabel} with {Math.Round(confidence * 100, 2)}%");
                        }
                    }

                    imageBoxFrameGrabber.Image = frame;
                }
            }
        }
        private void MainForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (capture != null)
                capture.Dispose();

            if (net != null)
                net.Dispose();
        }
    }
}
