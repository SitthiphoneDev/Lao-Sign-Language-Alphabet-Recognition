using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using System;
using System.Drawing;
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
            //faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");
            Application.Idle += ProcessFrame;

            // Load the ONNX model
            net = DnnInvoke.ReadNetFromONNX("\"C:\\Users\\Sitth\\source\\computer vision\\sign_lao_langude\\Model\\lsl_model.onnx\"");
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
                    CvInvoke.Flip(frame, frame, Emgu.CV.CvEnum.FlipType.Horizontal);

                    var grayFrame = frame.ToImage<Gray, byte>();
                    var faces = faceCascade.DetectMultiScale(grayFrame, 1.1, 3, new System.Drawing.Size(20, 20));

                    foreach (var face in faces)
                    {
                        var roiRect = face; // Use face directly, which is of type Rectangle
                        var roi = new Mat(frame, roiRect); // Create ROI using constructor
                        var avgVariance = CalculateAverageVariance(roi);

                        if (avgVariance > 1000)
                        {
                            var resizedRoi = new Mat();
                            CvInvoke.Resize(roi, resizedRoi, new System.Drawing.Size(100, 100), 0, 0, Emgu.CV.CvEnum.Inter.Linear);
                            var prediction = Predict(resizedRoi); // Pass Mat directly
                            var confidence = prediction.Max();
                            var predictedLabelIndex = Array.IndexOf(prediction, confidence);
                            var predictedLabel = labels[predictedLabelIndex];

                            Console.WriteLine($"Prediction: {predictedLabel} with {Math.Round(confidence * 100, 2)}%");
                        }
                    }

                    pictureBox1.Image = frame.ToBitmap();
                }
            }
        }

        private double[] Predict(Mat roi)
        {
            // Placeholder for your prediction code using Emgu.CV
            return new double[labels.Length];
        }


        private double CalculateAverageVariance(Mat roi)
        {
            var grayRoi = roi.ToImage<Gray, byte>(); // Convert Mat to Image<Gray, byte>
            var channels = grayRoi.Split(); // Split the grayscale image into channels

            var varianceSum = 0.0;

            for (int i = 0; i < channels.Length; i++)
            {
                var channel = channels[i];
                var mean = channel.GetAverage().Intensity; // Calculate the mean intensity of the channel

                double channelVariance = 0;

                // Calculate the variance of pixel intensities in the channel
                for (int x = 0; x < channel.Width; x++)
                {
                    for (int y = 0; y < channel.Height; y++)
                    {
                        var intensity = channel[y, x].Intensity;
                        channelVariance += Math.Pow(intensity - mean, 2);
                    }
                }

                channelVariance /= (channel.Width * channel.Height); // Normalize by the number of pixels
                varianceSum += channelVariance; // Accumulate the variance across channels
            }

            return varianceSum / channels.Length; // Return the average variance across channels
        }



        private double[] Predict(System.Drawing.Bitmap roi)
        {
            // Placeholder for your ONNX model prediction code
            return new double[labels.Length];
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
