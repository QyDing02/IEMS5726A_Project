const $file      = document.getElementById("file-input");
const $originImg = document.getElementById("origin-preview");
const $resultImg = document.getElementById("result-preview");
const $uploadBtn = document.getElementById("upload-btn");
const $clearBtn  = document.getElementById("clear-btn");
const $status    = document.getElementById("status");
const $outputSel = document.getElementById("output-type");

let selectedFile = null;

// ① 选文件 → 先本地预览
$file.addEventListener("change", (e) => {
  selectedFile = e.target.files[0];
  if (selectedFile) {
    $originImg.src = URL.createObjectURL(selectedFile);
    $originImg.hidden = false;
    $resultImg.hidden = true;          // 清掉上一次结果
    $status.textContent = "";
  }
});

// ② 点击上传并分析
$uploadBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    alert("Please choose an image first!");
    return;
  }
  const formData = new FormData();
  formData.append("file", selectedFile);

  $status.textContent = "⏳ Uploading and analyzing…";
  $uploadBtn.disabled = true;

  try {
    const res = await fetch(
      `http://192.168.31.39:5000/predict?output=${$outputSel.value}`,
      { method: "POST", body: formData }
    );
    const data = await res.json();

    if (data.status === "success") {
      $resultImg.src = `data:image/png;base64,${data.image}`;
      $resultImg.hidden = false;
      $status.textContent = "✅ Done";
    } else {
        $status.textContent = `❌ Server error: ${data.error || "Unknown error"}`;
    }
  } catch (err) {
    console.error(err);
    $status.textContent = "❌ Network error – is the backend running?";
  } finally {
    $uploadBtn.disabled = false;
  }
});

// ③ Clear 按钮：重置一切
$clearBtn.addEventListener("click", () => {
    if ($originImg.src) URL.revokeObjectURL($originImg.src); // 释放本地 URL

    selectedFile = null;
    $file.value = "";      // 清空 <input type="file">
    $originImg.hidden = true;
    $resultImg.hidden = true;
    $status.textContent = "";
});