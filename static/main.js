function fileSelected(file){
	var fileName = $(file).val();
	var selectButton = $("#select_button")[0];
	var previewDiv = $("#preview")[0];
	if (!fileName || !(fileName.endsWith(".jpg") || fileName.endsWith(".jpeg"))){
		selectButton.innerHTML = '<span style="color:red">Only JPEG images are supported</span>';
		previewDiv.innerHTML = "";
		return false;
	}
	
	selectButton.innerHTML = file.files[0].name;
	if(file.files && file.files[0]){
		var reader = new FileReader();
		reader.onload = function(evt){
			previewDiv.innerHTML = '<img alt="Selected Image" src="' + evt.target.result + '"/>';
		};
		reader.readAsDataURL(file.files[0]);
	}
}

$(document).ready(function(){
	$("#select_button").click(function(){
		$("input").click();
	});
	
	$("#submit").click(function(){
		var fileName = $("#file_uploader").val();
		var selectButton = $("#select_button")[0];
		if(!fileName){
			selectButton.innerHTML = '<span style="color:red">Please Select A Image</span>';
			return false;
		}
		if (!(fileName.endsWith(".jpg") || fileName.endsWith(".jpeg"))){
			alert("Only JPEG images are supported");
			return false;
		}
		
		var formData = new FormData();
		var file = $("#file_uploader")[0].files[0];
		formData.append("image_file", file);
		$.ajax({
			type: "POST",
			url: "/upload/",
			data: formData,
			processData: false,
			contentType: false,
			success: function(successData){
				jsonData = JSON.parse(successData);
				var resultTable = $("table");
				resultTable.css("display", "inline-block");
				var newTr = "<tr>";
				for(var i in jsonData){
					newTr += "<td>";
					newTr += jsonData[i];
					newTr += "</td>";
				}
				newTr += "</tr>";
				$("td").parent().remove();
				resultTable.append($(newTr));
			}
		});
	});
});