import axios from "axios";
import { useEffect, useRef, useState } from "react";
import "./App.css";
import PlaceHolder from "./assets/image_placeholder.png";

export default function App() {
  const [inputImageSrc, setInputImageSrc] = useState("");
  const [targetImageSrc, setTargetImageSrc] = useState("");
  const [uploadLoader, setUploadLoader] = useState(false);
  const [convertLoader, setConvertLoader] = useState(false);
  const [letter, setLetter] = useState("");
  const [font, setFont] = useState("");
  const [selectedFont, setSelectedFont] = useState("");
  const [allFonts, setAllFonts] = useState<{ [key: number]: string }>({});
  const [Matrix, setMatrix] = useState<number[][] | number[][][]>([]);
  const [Fontidx, setFontidx] = useState<number>(0);

  const [debug, setDebug] = useState(false);

  const input_REF = useRef<HTMLInputElement>(null);

  useEffect(() => {
    axios.get("http://localhost:5000/service/fonts").then((response: {
      data: {
        fonts: { [key: number]: string };
      }
    }) => {
      const fonts = response.data.fonts;
      // console.log(Object.keys(fonts));

      setAllFonts(fonts);
    });
  }, []);

  const handleUpload = async () => {
    try {
      let fileInput = input_REF.current;
      if (fileInput?.files && fileInput.files[0]) {
        const reader = new FileReader();
        reader.onload = async (e) => {
          const image = new Image();
          image.src = e.target?.result as string;
          image.onload = async () => {
            // Check if the image is less than 28x28
            // if (image.height > 28 || image.width > 28) {
            //   alert("Image size should be less than 28x28");
            //   return;
            // }
            setUploadLoader(true);
            if (fileInput.files) {
              setInputImageSrc(URL.createObjectURL(fileInput.files[0]));
            }
            const canvas = document.createElement("canvas");
            const context = canvas.getContext("2d");
            canvas.width = image.width;
            canvas.height = image.height;
            context?.drawImage(image, 0, 0);
            const ImageData = context?.getImageData(0, 0, image.width, image.height);
            const data = ImageData?.data;

            // convert the image data to a 2D array
            let matrix = [];
            for (let i = 0; i < image.height; i++) {
              let row = [];
              for (let j = 0; j < image.width; j++) {
                let index = (i * image.width + j) * 4;
                let r = data![index];
                let g = data![index + 1];
                let b = data![index + 2];
                // let a = data![index + 3];

                // convert the pixel to grayscale
                // let gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
                // row.push(gray);
                row.push([r, g, b]);
              }
              matrix.push(row);
            }
            setMatrix(matrix);
            // Send the matrix to the backend
            console.log("Sending matrix to backend");
            axios.post("http://localhost:5000/service/predict",{
              matrix
            }).then((response: { data: { font: string, letter: string} }) => {
              setLetter(response.data.letter);
              setFont(response.data.font);
              setUploadLoader(false);
            }
            );
          };
        }
        reader.readAsDataURL(fileInput.files[0]);
      }
    } catch (error) {
      console.error("Error uploading image:", error);
    }
  };

  const handleGenerate = async () => {
    setConvertLoader(true);
    setTargetImageSrc("");
    // Send the matrix to the backend
    axios.post("http://localhost:5000/service/generate", {
      letter: letter,
      font_idx: Fontidx,
      matrix: Matrix,
    }).then((response) => {

      // Get the Base64 image from the response
      const base64Image = response.data.image;

      // Set the Base64 image as the source for an <img> tag
      setTargetImageSrc(`data:image/png;base64,${base64Image}`);
      setConvertLoader(false);
    }).catch((error) => {
      console.error("Error converting matrix to image:", error);
      setConvertLoader(false);
    });
  };
  return (
    <>
    {/* toggle debug mode */}
    <button onClick={() => setDebug(!debug)}>{debug ? "Debug Mode is Active" : "Debug Mode"}</button>
    <main>
      <div>
        <h1>Input Image</h1>
        {inputImageSrc && <img src={inputImageSrc} alt="Input Image" /> || uploadLoader && <ImageLoading /> || <ImagePlaceholder />}
        <input type="file" ref={input_REF} onChange={handleUpload} accept="image/*" />
      </div>
      <div>
        <h1>Output Image</h1>
        {targetImageSrc && <img src={targetImageSrc} alt="Ouput Image" /> || convertLoader && <ImageLoading /> || <ImagePlaceholder />}
      </div>
      <div className="input_details">
        <button onClick={()=> input_REF.current?.click()}>Upload Image</button>
        <p>Font: {font && font || "NA"}</p>
        <select value={selectedFont} onChange={(e) => {setSelectedFont(e.target.value); setFontidx(parseInt(e.target.value));}}>
          <option value="" style={{display:"none"}} >Select Font</option>
          {Object.keys(allFonts).map((key: string) => (
            <option key={key} value={key}>{allFonts[parseInt(key)]}</option>
          ))}
        </select>
        <button onClick={handleGenerate}>Convert</button>
      </div>
    </main>
    {
      debug ?
      <textarea value={letter} onChange={(e) => setLetter(e.target.value)}></textarea>
      : <p>Text: {letter && letter || "NA"}</p>
      }
    </>
  );
  // // create a 2D array of random numbers 128x128
  // const matrix = Array.from({ length: 128 }, () =>
  //   Array.from({ length: 128 }, () => Math.floor(Math.random() * 256))
  // );


  // const handleConvert = async () => {
  //   try {
  //     // Send the matrix to the backend
  //     const response = await axios.post("http://localhost:5000/convert/matrix2img", {
  //       matrix,
  //     });

  //     // Get the Base64 image from the response
  //     const base64Image = response.data.image;

  //     const response2 = await axios.post("http://localhost:5000/convert/img2matrix", {
  //       image: base64Image,
  //     });

  //     // Get the matrix from the response
  //     const matrix2 = response2.data.matrix;

  //     // check if the matrix is the same as the original matrix
  //     let  equal = JSON.stringify(matrix) === JSON.stringify(matrix2);
  //     console.log(equal);

  //     // Set the Base64 image as the source for an <img> tag
  //     setImageSrc(`data:image/png;base64,${base64Image}`);
  //   } catch (error) {
  //     console.error("Error converting matrix to image:", error);
  //   }
  // };

};


function ImagePlaceholder() {
  return (
    <img src={PlaceHolder} alt="Placeholder"/>
  );
}

function ImageLoading() {
  return (
    <div className="img loading_container">
      <div className="loading"></div>
    </div>
  );
}