document.getElementById('submit-button').addEventListener('click', () => {
    comment = document.getElementById('commentaire-input').value

    // Send request python API
    protocol = 'http://'
    host = '127.0.0.1'
    port = ':5000'
    path = '/check-comment'
    url = protocol + host + port + path

    axios.post(url, {
      comment: comment
    })
    .then(function (response) {
      console.log(response);
      console.log(response.data.status)
      // Receive response
      if (response.status != 200) {
        throw new Error(`HTTP error! status: ${response.data.status}`);
      } else {
          switch (response.data.status) {
            case 'positive':
              document.getElementById('status').textContent = "Merci mon pote !"
              document.getElementById('status').style.color = "green";
              break;
            case 'negative':
              document.getElementById('status').textContent = "Pas très gentil tout ça !"
              document.getElementById('status').style.color = "orange";
              break
            case 'suicidal':
              document.getElementById('status').textContent = "Super l'ambiance ..."
              document.getElementById('status').style.color = "red";
            default:
              console.error('NO STATUS !!!')
              break;
          }
      }
    })
    .catch(function (error) {
      console.log(error);
    });
});