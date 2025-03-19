const {initializeApp} = require('firebase/app');
const {getFirestore,doc, setDoc, vector} = require("firebase/firestore")
const {getDatabase,ref,get} = require("firebase/database")
const {GoogleGenerativeAI} = require("@google/generative-ai")
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const  {GoogleGenerativeAIEmbeddings} = require("@langchain/google-genai");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter")
const {getStorage, ref:storageRef,uploadBytes,getDownloadURL} = require("firebase/storage")
const fs = require("fs")
const path = require("path")
require("dotenv").config()

const firebaseConfig = {
    apiKey: process.env.apiKey,
    authDomain: process.env.authDomain,
    databaseURL: process.env.databaseURL,
    projectId: process.env.projectId,
    storageBucket: process.env.storageBucket,
    messagingSenderId: process.env.messagingSenderId,
    appId: process.env.appId,
    measurementId: process.env.measurementId
};



const geminiApiKey = process.env.GOOGLE_API_KEY ;

const genAI = new GoogleGenerativeAI(geminiApiKey);
const model = genAI.getGenerativeModel({model:"gemini-pro"});


const app = initializeApp(firebaseConfig)
const firestore = getFirestore(app);
const storage = getStorage(app)
const db = getDatabase(app)


async function redactorPII(data){
  const regexs = {
    emailRegex: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/g,
    phoneRegex: /\b(?:\+?1\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/g,
    ssnRegex: /\b\d{3}-\d{2}-\d{4}\b/g,
    creditCardRegex: /\b(4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b/g,
    addressRegex: /\d{1,5}\s\w+(\s\w+)?\s(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Parkway|Pkwy|Circle|Cir|Terrace|Ter)\b/g,
    passportRegex: /\b[A-Z]{2}\d{7}\b/g,
    ipAddressRegex: /\b((25[0-5]|2[0-4][0-9]|1?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|1?[0-9][0-9]?)\b/g,
    driversLicenseRegex: /\b[A-Z0-9]{5,20}\b/g,
    usernameRegex: /^[a-zA-Z0-9._-]{3,20}$/g,
    passwordRegex: /^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/g
  }

  for (const key in regexs){
    data = data.replace(regexs[key],"[REDACTED]")
  }

  return data

}



async function loadDataFromDatabase(path){
    const dbRef = ref(db, path);
  try {
    const snapshot = await get(dbRef);
    if (snapshot.exists()) {
      const items = snapshot.val();
      const documents = [];
      for(const data of items){
        const conversations = data.conversations || []; 
      const initial_query = data.initial_query || "";
      const product_title = data.product_title || "";
      const ticket_created_at = data.ticket_created_at || "";
      const ticket_id = data.ticket_id || "";

      documents.push({
        id: `ticket-${ticket_id}`,
        text: initial_query,
        metadata: {
          product_title: product_title,
          ticket_created_at: ticket_created_at,
          type: "initial_query"
        }
      });

      conversations.forEach((conversation, index) => {
        documents.push({
          id: `ticket-${ticket_id}-conversation-${index}`,
          text: conversation.conversation_content,
          metadata: {
            ticket_id: ticket_id,
            conversation_created_at: conversation.conversation_created_at,
            type: "conversation"
          }
        });
      });
      }
      console.log("documents:",documents)
      return documents;
    } else {
      console.log("No data available at the specified path.");
      return [];
    }
  } catch (error) {
    console.error("Error fetching data from Realtime Database:", error);
    return [];
  }
}

let  a =true;
async function createEmbeddings(documents, batchSize = 1000) {
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 0,
    });

    let vectorStores = null;
    const docs = [];
        for (const doc of documents) {
            const redactedText = await redactorPII(doc.text)
            if(a){
              console.log("redacted",redactedText);
              console.log("orginal",doc.text)
              a = false;
            }
            const splits = await textSplitter.splitText(redactedText);
            for (const split of splits) {
                docs.push({
                    pageContent: split,
                    metadata: { ...doc.metadata, doc_id: doc.id }
                });
            }
        }

        console.log(`Processing ${docs.length} total document chunks`);
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: geminiApiKey,
      modelName: "embedding-001"
  });
    for (let i = 0; i < docs.length; i += batchSize) {
        console.log(`Processing batch ${i / batchSize + 1}`);
        const batchDocs = docs.slice(i, i + batchSize);
        const batchVectorStore = await FaissStore.fromDocuments(batchDocs,embeddings);

        if(!vectorStores){
          vectorStores = batchVectorStore
        }else{
          const tempDir = "./faiss_temp_batch";
          await batchVectorStore.save(tempDir);

          const batchIndex = await FaissStore.load(tempDir,embeddings)

          await vectorStores.mergeFrom(batchIndex);

          fs.rmSync(tempDir,{recursive:true,force:true})
        }
        console.log(`Batch ${Math.floor(i/batchSize) + 1} processed and merged`)   
    }

    return vectorStores;
}

async function storeFaissIndexToFireBase(vectorStore, folderName, fileName) {
    const tempDir = './faiss_temp';
    await vectorStore.save(tempDir);

    const indexBuffer = fs.readFileSync(path.join(tempDir, 'faiss.index'));
    const docstoreBuffer = fs.readFileSync(path.join(tempDir, 'docstore.json'));

    const indexRef = storageRef(storage, `${folderName}/${fileName}-faiss.index`);
    const docstoreRef = storageRef(storage, `${folderName}/${fileName}-docstore.json`);
    await uploadBytes(indexRef, indexBuffer);
    await uploadBytes(docstoreRef, docstoreBuffer);


    const indexURL = await getDownloadURL(indexRef);
    const docstoreURL = await getDownloadURL(docstoreRef);

    console.log("FAISS index stored in Firebase Storage");

    fs.rmSync(tempDir, { recursive: true, force: true });

    return { indexURL, docstoreURL }
}

async function runMain(){
    try {
        const documents = await loadDataFromDatabase("/")

        const vectorStores = await createEmbeddings(documents, 2000);

       const {indexURL, docstoreURL }= await storeFaissIndexToFireBase(
        vectorStores,
        'faiss_indexes',
        'combined-index'
       )

       const docRef = doc(firestore,'faiss_indexes','combined-index')
       await setDoc(docRef,{
        indexURL,
        docstoreURL,
        documentCount: documents.length,
       })
       console.log("combined FAISS index URLS stored in firestore")
    } catch (error) {
        console.error("Error:", error);
    }
}

runMain()