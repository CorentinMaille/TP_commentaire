document.onclick(() => {
    // Send request python API
    fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
      .then(response => {
        // Receive response
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        } else {
            if (response.body.positive) {
                // If positive comment
                document.getElementById('status').textContent = "POSITIF !!!"
                document.getElementById('status').style.color = "green";
            } else {
                // If negative comment
                document.getElementById('status').textContent = "NEGATIF !!!"
                document.getElementById('status').style.color = "red";
            }
        }
    })
      .catch(error => {
        // Handle any errors that occurred during the fetch
        console.error('There was an error!', error);
    });


});