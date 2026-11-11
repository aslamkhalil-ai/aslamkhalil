async function copyText(id){
  const el=document.getElementById(id);
  if(!el)return;
  try{await navigator.clipboard.writeText(el.innerText);alert("Copied to clipboard.");}
  catch(e){alert("Clipboard access unavailable. Please select and copy the text manually.");}
}
