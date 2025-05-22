package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/disintegration/imaging"
	"github.com/gorilla/mux"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/mongo/readpref"
)

// ========== Global Setup ==========
var uploadPath = "./uploads"
var mongoClient *mongo.Client
var bacteriaCollection *mongo.Collection

func init() {
	os.MkdirAll(uploadPath, os.ModePerm)
}

// ========== Image Handling ==========

func loadUploadedImg() (image.Image, error) {
	img, err := imaging.Open(filepath.Join(uploadPath, "image.jpg"))
	if err != nil {
		return nil, err
	}
	return img, nil
}

func saveOutputImg(w http.ResponseWriter, img image.Image) {
	outputPath := filepath.Join(uploadPath, "output.jpg")
	err := imaging.Save(img, outputPath)
	if err != nil {
		http.Error(w, "Failed to save image", http.StatusInternalServerError)
		return
	}
	w.Write([]byte("Image processed and saved to output.jpg"))
}

func uploadHandler(w http.ResponseWriter, r *http.Request) {
	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "Unable to read image", http.StatusBadRequest)
		return
	}
	defer file.Close()

	dst, err := os.Create(filepath.Join(uploadPath, "image.jpg"))
	if err != nil {
		http.Error(w, "Unable to save image", http.StatusInternalServerError)
		return
	}
	defer dst.Close()

	_, err = io.Copy(dst, file)
	if err != nil {
		http.Error(w, "Failed to copy image", http.StatusInternalServerError)
		return
	}
	w.Write([]byte("Image uploaded successfully"))
}

// ========== Image Transformations ==========

func grayscaleHandler(w http.ResponseWriter, r *http.Request) {
	img, err := loadUploadedImg()
	if err != nil {
		http.Error(w, "Failed to load image", http.StatusInternalServerError)
		return
	}
	saveOutputImg(w, imaging.Grayscale(img))
}

func rotateHandler(w http.ResponseWriter, r *http.Request) {
	img, _ := loadUploadedImg()
	angle, _ := strconv.ParseFloat(r.URL.Query().Get("angle"), 64)
	saveOutputImg(w, imaging.Rotate(img, angle, color.White))
}

func mirrorHandler(w http.ResponseWriter, r *http.Request) {
	img, _ := loadUploadedImg()
	saveOutputImg(w, imaging.FlipH(img))
}

func blurHandler(w http.ResponseWriter, r *http.Request) {
	img, _ := loadUploadedImg()
	radius, _ := strconv.ParseFloat(r.URL.Query().Get("radius"), 64)
	saveOutputImg(w, imaging.Blur(img, radius))
}

func sharpenHandler(w http.ResponseWriter, r *http.Request) {
	img, _ := loadUploadedImg()
	amount, _ := strconv.ParseFloat(r.URL.Query().Get("amount"), 64)
	saveOutputImg(w, imaging.Sharpen(img, amount))
}

func resizeHandler(w http.ResponseWriter, r *http.Request) {
	img, _ := loadUploadedImg()
	width, _ := strconv.Atoi(r.URL.Query().Get("width"))
	height, _ := strconv.Atoi(r.URL.Query().Get("height"))
	saveOutputImg(w, imaging.Resize(img, width, height, imaging.Lanczos))
}

func cropHandler(w http.ResponseWriter, r *http.Request) {
	img, _ := loadUploadedImg()
	width, _ := strconv.Atoi(r.URL.Query().Get("width"))
	height, _ := strconv.Atoi(r.URL.Query().Get("height"))
	saveOutputImg(w, imaging.CropCenter(img, width, height))
}

func brightnessHandler(w http.ResponseWriter, r *http.Request) {
	img, _ := loadUploadedImg()
	level, _ := strconv.ParseFloat(r.URL.Query().Get("level"), 64)
	saveOutputImg(w, imaging.AdjustBrightness(img, level))
}

func contrastHandler(w http.ResponseWriter, r *http.Request) {
	img, _ := loadUploadedImg()
	level, _ := strconv.ParseFloat(r.URL.Query().Get("level"), 64)
	saveOutputImg(w, imaging.AdjustContrast(img, level))
}

func edgeHandler(w http.ResponseWriter, r *http.Request) {
	src, _ := loadUploadedImg()
	bounds := src.Bounds()
	dst := image.NewGray(bounds)

	Gx := [3][3]int{
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1},
	}
	Gy := [3][3]int{
		{-1, -2, -1},
		{0, 0, 0},
		{1, 2, 1},
	}

	for y := 1; y < bounds.Max.Y-1; y++ {
		for x := 1; x < bounds.Max.X-1; x++ {
			var gx, gy int
			for j := -1; j <= 1; j++ {
				for i := -1; i <= 1; i++ {
					r, _, _, _ := src.At(x+i, y+j).RGBA()
					gray := int(r >> 8)
					gx += gray * Gx[j+1][i+1]
					gy += gray * Gy[j+1][i+1]
				}
			}
			mag := uint8(math.Min(255, math.Sqrt(float64(gx*gx+gy*gy))))
			dst.SetGray(x, y, color.Gray{Y: mag})
		}
	}
	saveOutputImg(w, dst)
}

// ========== TCP Server ==========

type BacteriaData struct {
	Image         string `json:"image" bson:"image"`
	Prediction    string `json:"prediction" bson:"prediction"`
	EncryptionKey string `json:"encryption_key" bson:"encryption_key"`
}

var dataStore = make(map[string]BacteriaData)

func saveBase64Image(b64Str string, filename string) error {
	data, err := base64.StdEncoding.DecodeString(b64Str)
	if err != nil {
		return err
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return err
	}
	outFile, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer outFile.Close()
	return jpeg.Encode(outFile, img, nil)
}

func handleTCPConnection(conn net.Conn) {
	defer conn.Close()

	var receivedData BacteriaData
	decoder := json.NewDecoder(conn)
	err := decoder.Decode(&receivedData)
	if err != nil {
		fmt.Println("Error decoding data:", err)
		return
	}

	// Save to in-memory store
	dataStore[receivedData.EncryptionKey] = receivedData

	// Save image to file
	imgPath := filepath.Join(uploadPath, "tcp_image.jpg")
	err = saveBase64Image(receivedData.Image, imgPath)
	if err != nil {
		fmt.Println("Failed to save image:", err)
		conn.Write([]byte("Failed to save image"))
		return
	}

	// Save to MongoDB
	_, err = bacteriaCollection.InsertOne(context.TODO(), receivedData)
	if err != nil {
		fmt.Println("Failed to insert into MongoDB:", err)
	}

	fmt.Println("Data received and image saved:", receivedData.EncryptionKey)
	conn.Write([]byte("Data received and image saved successfully"))
}

func startTCPServer() {
	listener, err := net.Listen("tcp", ":12345")
	if err != nil {
		fmt.Println("Error starting TCP server:", err)
		return
	}
	defer listener.Close()

	fmt.Println("TCP Server listening on port 12345...")
	for {
		conn, err := listener.Accept()
		if err != nil {
			fmt.Println("Error accepting TCP connection:", err)
			continue
		}
		go handleTCPConnection(conn)
	}
}

// ========== HTTP Views ==========

func viewTCPImageHandler(w http.ResponseWriter, r *http.Request) {
	imagePath := filepath.Join(uploadPath, "tcp_image.jpg")
	http.ServeFile(w, r, imagePath)
}

func getPredictionsHandler(w http.ResponseWriter, r *http.Request) {
	cursor, err := bacteriaCollection.Find(context.TODO(), bson.M{})
	if err != nil {
		http.Error(w, "MongoDB query failed", http.StatusInternalServerError)
		return
	}
	defer cursor.Close(context.TODO())

	var results []BacteriaData
	if err := cursor.All(context.TODO(), &results); err != nil {
		http.Error(w, "Failed to parse MongoDB results", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

// ========== Main ==========

func main() {
	// Start TCP server
	go startTCPServer()

	// Connect to MongoDB
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var err error
	mongoClient, err = mongo.Connect(ctx, options.Client().ApplyURI("mongodb+srv://sgshivapalaksha:phGTYpu27cidpgVh@mycluster0.hx7xo.mongodb.net/"))
	if err != nil {
		log.Fatal("MongoDB connection error:", err)
	}
	if err := mongoClient.Ping(ctx, readpref.Primary()); err != nil {
		log.Fatal("MongoDB ping failed:", err)
	}
	bacteriaCollection = mongoClient.Database("bacteriaDB").Collection("predictions")
	fmt.Println("Connected to MongoDB")

	// Setup HTTP routes
	router := mux.NewRouter()

	router.HandleFunc("/upload", uploadHandler).Methods("POST")
	router.HandleFunc("/grayscale", grayscaleHandler).Methods("POST")
	router.HandleFunc("/rotate", rotateHandler).Methods("POST")
	router.HandleFunc("/mirror", mirrorHandler).Methods("POST")
	router.HandleFunc("/blur", blurHandler).Methods("POST")
	router.HandleFunc("/sharpen", sharpenHandler).Methods("POST")
	router.HandleFunc("/resize", resizeHandler).Methods("POST")
	router.HandleFunc("/crop", cropHandler).Methods("POST")
	router.HandleFunc("/brightness", brightnessHandler).Methods("POST")
	router.HandleFunc("/contrast", contrastHandler).Methods("POST")
	router.HandleFunc("/edge", edgeHandler).Methods("POST")
	router.HandleFunc("/view-tcp-image", viewTCPImageHandler).Methods("GET")
	router.HandleFunc("/predictions", getPredictionsHandler).Methods("GET")

	log.Println("HTTP Server is running at http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", router))
}
