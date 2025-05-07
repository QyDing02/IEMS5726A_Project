const $file      = document.getElementById("file-input");
const $originImg = document.getElementById("origin-preview");
const $resultImg = document.getElementById("result-preview");
const $uploadBtn = document.getElementById("upload-btn");
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
    alert("请先选择一张图片！");
    return;
  }
  const formData = new FormData();
  formData.append("file", selectedFile);

  $status.textContent = "⏳ 正在上传并分析...";
  $uploadBtn.disabled = true;

  try {
    const res = await fetch(
      `http://localhost:5000/predict?output=${$outputSel.value}`,
      { method: "POST", body: formData }
    );
    const data = await res.json();

    if (data.status === "success") {
      $resultImg.src = `data:image/png;base64,${data.image}`;
      $resultImg.hidden = false;
      $status.textContent = "✅ 分析完成";
    } else {
      $status.textContent = `❌ 服务器错误：${data.error || "未知错误"}`;
    }
  } catch (err) {
    console.error(err);
    $status.textContent = "❌ 网络错误，请检查后端是否启动";
  } finally {
    $uploadBtn.disabled = false;
  }
});
