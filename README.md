### Category Identifier

For the given unstructured or semi structured data of document,
grouped labels are identified.

Steps:
---

* clone the repository

        https://github.com/arohini/CategoryFinder.git

* Install the requirements

        pip install -r requirements.txt
        
* To train the data

        python train_topic_modeling.py
     
   Note: For the resulted tuple of topic_num and topics,
   replace topic num with the label which is most suited 
   and use the same in findtopic.py

* Test the trained data

        python find_topic.py