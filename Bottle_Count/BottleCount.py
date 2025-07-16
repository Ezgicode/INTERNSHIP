
    import cv2
    import os 
    
    def bottles_numbers(full_path):
        image = cv2.imread(full_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(gray,(3,3),0)
        blur = gray
        _, adaptresh= cv2.threshold (blur,125,255,cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(adaptresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
        #çerçeveyi sil
        areas = [cv2.contourArea(cnt) for cnt in contours]
        max_area = max(areas)
    
    
        number_bottles = 0
        for num in contours:
            area = cv2.contourArea(num)
            perimeter = cv2.arcLength(num, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * 3.14 * area / (perimeter * perimeter)
            
            if 4000> area > 30 and 0.3 < circularity <= 1.3:
                cv2.drawContours(image, [num], -1, (0, 0, 255), 2)  # sayılanlar (kırmızı)
                number_bottles +=1
                #print(f"Area: {area}")

            else:
                cv2.drawContours(image, [num], -1, (255, 0, 0), 2)  # diğer mavi
                #print(f"Area: {area} bu sayılmıyor")
    #bozukkonrulerigösteramakafamıkarıştırdı
    
        
        cv2.imshow(f"Konturlar : {os.path.basename(full_path)}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
      
        return number_bottles
    
    
        
    
    path = r"C:\Users\ezgii\OneDrive\Resimler\images"
    
    for img in os.listdir(path):
        if img.lower().endswith(".png"):
            full_path = os.path.join(path,img)
            numbers = bottles_numbers(full_path)
            print(f"{img}: {numbers} şişe")       
            print(full_path)
